import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
import mmcv
import PIL
import cv2
import xml.etree.ElementTree as ET
import torch
import numpy as np
import json
import time
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class FeatureEnhancementTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, num_layers=4):
        super(FeatureEnhancementTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.embedding = nn.Linear(32, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MyReIDModel(nn.Module):
    def __init__(self, feature_dim=768):
        super(MyReIDModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # 输出固定 (8,8)
        )
        self.fc = None
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.fc is None:
            in_features = x.size(1)
            self.fc = nn.Linear(in_features, self.feature_dim).to(x.device)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def extract_reid_features(model, image, bboxes):
    if len(bboxes) == 0:
        return np.empty((0, 768))
    transform = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    crops = []
    for box in bboxes:
        x1, y1, x2, y2 = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        crop = image[y1:y2, x1:x2, :]
        if crop.size == 0:
            crop = np.zeros((256,128,3), dtype=np.uint8)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)
        crop = transform(crop)
        crops.append(crop)
    crops = torch.stack(crops, dim=0)
    crops = crops.to(next(model.parameters()).device)
    with torch.no_grad():
        feats = model(crops)
        feats = F.normalize(feats, p=2, dim=1)
    return feats.cpu().numpy()


class LSTMTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=1, seq_len=5, pred_len=3):
        super(LSTMTrajectoryPredictor, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 线性映射将2维 (x,y) -> hidden_dim
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim * pred_len)

    def forward(self, traj):
        """
        traj: [batch_size, seq_len, 2]
        返回 [batch_size, pred_len, 2]
        """
        bsz = traj.size(0)
        x = self.input_embedding(traj)       # [bsz, seq_len, hidden_dim]
        x, _ = self.lstm(x)                  # [bsz, seq_len, hidden_dim]
        x = x[:, -1]                         # 取最后时刻输出
        out = self.head(x)                   # [bsz, pred_len*2]
        return out.view(bsz, self.pred_len, 2)


# 存储: predictions_store[(id, future_frame)] = (px, py)
# 在某一帧做预测后，会把id, future_frame(=当前帧+1..+pred_len) 对应的预测中心记录下来
predictions_store = {}  # {(id, frame_i): (x, y)}

# 存储误差：errors_list = [(frame_idx, id, horizon, error_dist), ...]
errors_list = []

def compute_and_store_error_for_frame(frame_idx, track_bboxes):
    """
    在处理完第 frame_idx 帧时，查看是否存在先前帧对 frame_idx 的预测。
    如果存在，就计算 (预测坐标 vs 本帧的真实中心) 的误差，并存入 errors_list。
    track_bboxes: 当前帧实际检测/跟踪到的目标 [N, 6], 其中 [:,0] = ID
    """
    for bbox in track_bboxes:
        _id = int(bbox[0])
        x1, y1, x2, y2 = bbox[1:5]
        cx_real = (x1 + x2) / 2
        cy_real = (y1 + y2) / 2
        # 如果在 predictions_store 中找到了 (id, frame_idx) 的预测值
        if (_id, frame_idx) in predictions_store:
            (px, py, predict_from_frame) = predictions_store[(_id, frame_idx)]
            # 计算误差
            dist = np.sqrt((px - cx_real)**2 + (py - cy_real)**2)
            horizon = frame_idx - predict_from_frame  # 例如：frame_idx=10, predict_from_frame=9 => horizon=1
            # 存入全局 errors_list
            errors_list.append((frame_idx, _id, horizon, float(dist)))
            # 用完这条预测后，可以删掉
            # （如果你想保留以便可视化，也可以选择不删除）
            del predictions_store[(_id, frame_idx)]

def plot_and_save_all_errors(save_dir):
    """
    在所有帧处理完毕后，基于全局 errors_list 做可视化绘图并保存。
    同时会打印误差统计：平均/中位/最大误差，以及ADE/FDE等。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(errors_list) == 0:
        print("[Warning] No prediction errors to plot. Did you predict any future frames?")
        return

    # 将 errors_list 转成 np.array 便于统计
    # shape: (num_samples, 4) => [frame_idx, id, horizon, dist]
    arr = np.array(errors_list)
    dists = arr[:, 3]  # 全部误差
    horizons = arr[:, 2]

    mean_error = np.mean(dists)
    median_error = np.median(dists)
    max_error = np.max(dists)

    print("======= 误差统计（所有预测步） =======")
    print(f"平均误差 (Mean Error) = {mean_error:.3f}")
    print(f"中位数误差 (Median Error) = {median_error:.3f}")
    print(f"最大误差 (Max Error) = {max_error:.3f}")
    print("===================================")

    # 1) 画一个所有误差的直方图
    plt.figure()
    plt.title("All Prediction Errors Histogram")
    plt.xlabel("Error Distance")
    plt.ylabel("Frequency")
    plt.hist(dists, bins=30)  # 不指定颜色
    plt.savefig(os.path.join(save_dir, "all_errors_hist.png"))
    plt.close()

    # 2) 每帧的平均误差折线图
    #    需要先按frame_idx分组
    unique_frames = np.unique(arr[:, 0])
    frame_avg_errors = []
    for fidx in unique_frames:
        fmask = (arr[:, 0] == fidx)
        frame_avg_errors.append((fidx, np.mean(arr[fmask, 3])))
    frame_avg_errors = sorted(frame_avg_errors, key=lambda x: x[0])

    plt.figure()
    plt.title("Average Prediction Error per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Error")
    frames_ = [x[0] for x in frame_avg_errors]
    errs_ = [x[1] for x in frame_avg_errors]
    plt.plot(frames_, errs_)
    plt.savefig(os.path.join(save_dir, "frame_avg_error_line.png"))
    plt.close()


    horizon_1_errors = arr[horizons == 1, 3]
    horizon_2_errors = arr[horizons == 2, 3]
    horizon_3_errors = arr[horizons == 3, 3]

    ade_1 = np.mean(horizon_1_errors) if len(horizon_1_errors)>0 else 0
    ade_2 = np.mean(horizon_2_errors) if len(horizon_2_errors)>0 else 0
    ade_3 = np.mean(horizon_3_errors) if len(horizon_3_errors)>0 else 0


    simple_ade = np.mean([ade_1, ade_2, ade_3])
    simple_fde = np.mean(horizon_3_errors) if len(horizon_3_errors)>0 else 0

    print(f"[Simple Demo] ADE (3步平均) = {simple_ade:.3f}, FDE (最后一步) = {simple_fde:.3f}")


    plt.figure()
    plt.title("Horizon-wise Error (ADE/FDE Example)")
    plt.xlabel("Horizon")
    plt.ylabel("Mean Error")
    horizons_x = [1, 2, 3]
    means_ = [ade_1, ade_2, ade_3]
    plt.plot(horizons_x, means_, marker='o')
    # 在图中写一下ADE,FDE
    plt.text(1.8, np.max(means_)*0.8, f"ADE~{simple_ade:.3f}\nFDE~{simple_fde:.3f}")
    plt.savefig(os.path.join(save_dir, "ADE_FDE_example.png"))
    plt.close()

    # 4) 给每个目标id的误差随帧数变化画图
    #    如果ID很多，这个图会很多，可以只画若干个或者任选
    unique_ids = np.unique(arr[:, 1])
    for uid in unique_ids[:5]:  # 演示前5个目标
        mask = (arr[:, 1] == uid)
        arr_id = arr[mask]
        frames_id = arr_id[:, 0]
        # 也可以按horizon分开画
        frame_error_dict = {}
        for row in arr_id:
            f_, _, _, dist_ = row
            if f_ not in frame_error_dict:
                frame_error_dict[f_] = []
            frame_error_dict[f_].append(dist_)
        sorted_items = sorted(frame_error_dict.items(), key=lambda x: x[0])
        frames_ = [k for (k,v) in sorted_items]
        errs_ = [np.mean(v) for (k,v) in sorted_items]

        plt.figure()
        plt.title(f"ID={int(uid)} Error over Frames")
        plt.xlabel("Frame Index")
        plt.ylabel("Mean Error (this ID)")
        plt.plot(frames_, errs_)
        plt.savefig(os.path.join(save_dir, f"id_{int(uid)}_error_line.png"))
        plt.close()


##########################################
# 3. 更新轨迹并可视化预测 + 存储预测结果
##########################################
def update_and_predict(traj_history, track_bboxes, image, predictor, color=(0,255,0), frame_idx=None):
    """
    traj_history: dict, key=目标ID, value=历史中心点列表
    track_bboxes: [N, 6] => [id, x1, y1, x2, y2, score]
    image: 当前帧图像
    predictor: 轨迹预测模型(LSTM)
    color: 画线颜色
    frame_idx: 当前是第几帧，用于后续误差计算时记录
    """
    seq_len = predictor.seq_len
    pred_len = predictor.pred_len
    img_h, img_w = image.shape[:2]

    for bbox in track_bboxes
        _id, x1, y1, x2, y2 = bbox[:5].astype(int)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # 更新轨迹
        if _id not in traj_history:
            traj_history[_id] = []
        traj_history[_id].append([cx, cy])
        if len(traj_history[_id]) > seq_len:
            traj_history[_id].pop(0)

        # 只有当历史帧数达到 seq_len，才预测
        if len(traj_history[_id]) == seq_len:
            # 构造输入
            input_seq = torch.tensor([traj_history[_id]], dtype=torch.float32)  # [1, seq_len, 2]
            device = next(predictor.parameters()).device
            input_seq = input_seq.to(device)

            with torch.no_grad():
                pred_out = predictor(input_seq)[0].cpu().numpy()  # [pred_len, 2]

            # Debug打印
            print(f"[Debug] Frame={frame_idx}, ID={_id}, Predicted offsets=\n{pred_out}")

            # 在图像上可视化
            prev_pt = (int(cx), int(cy))
            for step_i, (px, py) in enumerate(pred_out, start=1):
                # clip 防止坐标越界
                px = int(np.clip(px, 0, img_w - 1))
                py = int(np.clip(py, 0, img_h - 1))
                cv2.circle(image, (px, py), 3, color, -1)
                cv2.line(image, prev_pt, (px, py), color, 2)
                prev_pt = (px, py)

            # ======== 存储到 predictions_store 以便后续帧做误差计算 ========
            # 假设本帧是 frame_idx，那么预测出 frame_idx+1.. frame_idx+pred_len
            if frame_idx is not None:
                for horizon_i in range(pred_len):
                    future_frame = frame_idx + horizon_i + 1
                    px, py = pred_out[horizon_i]
                    # 记下 (px, py) 以及是从哪帧预测过来的
                    predictions_store[(_id, future_frame)] = (px, py, frame_idx)


##########################################
# 4. 多视角的各种匹配/ID合并函数
##########################################
def matching(leftImage, rightImage):
    kp1, kp2, matches, matchesMask = create_sift(leftImage, rightImage)
    good, matchesMask, M = compute_matrics(matches, kp1, kp2)
    return M

def create_sift(leftImage, rightImage):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(leftImage, None)
    kp2, des2 = sift.detectAndCompute(rightImage, None)
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    return kp1, kp2, matches, matchesMask

def compute_matrics(matches, kp1, kp2):
    MIN_MATCH_COUNT = 10
    good = []
    query_matched = []
    train_matched = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            if m.queryIdx not in query_matched and m.trainIdx not in train_matched:
                query_matched.append(m.queryIdx)
                train_matched.append(m.trainIdx)
                good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        M = 0
        matchesMask = None
    return good, matchesMask, M

def compute_transf_matrix(pts_src, pts_dst, f_last, image11, image22):
    if len(pts_src) >= 5:
        f, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
        if f is None:
            f = f_last.copy()
        f_last = f.copy()
    else:
        M = matching(image11, image22)
        f = M
        if isinstance(f, int) or f is None or (isinstance(f, np.ndarray) and f.all() == 0):
            f = f_last.copy()
        f_last = f.copy()
    return f, f_last

def all_nms(dets, thresh):
    if len(dets) == 0:
        return dets
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    scores = dets[:, 5]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = np.array([]).reshape(0, dets.shape[1])
    while order.size > 0:
        i = order[0]
        keep = np.vstack((keep, dets[i]))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def read_xml_r(xml_file1, i):
    root1 = ET.parse(xml_file1).getroot()
    track_all = root1.findall('track')
    bboxes1 = []
    ids = []
    id = 0
    for track in track_all:
        id_label = int(track.attrib['id'])
        boxes = track.findall('box')
        shape = [1080, 1920]
        for box in boxes:
            if int(box.attrib['frame']) == i:
                xtl = int(float(box.attrib['xtl']))
                ytl = int(float(box.attrib['ytl']))
                xbr = int(float(box.attrib['xbr']))
                ybr = int(float(box.attrib['ybr']))
                outside = int(box.attrib['outside'])
                if outside == 1 or xtl <= 10 and ytl <= 10 or xbr >= shape[1] - 0 and ytl <= 0:
                    break
                confidence = 0.99
                bboxes1.append([xtl, ytl, xbr, ybr, 1])
                ids.append(id_label)
                id += 1
                break
    if len(bboxes1) == 0:
        bboxes1 = np.array([]).reshape(0, 5)
        ids = np.array([])
        labels = np.array([])
    else:
        bboxes1 = torch.tensor(bboxes1, dtype=torch.long)
        ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.zeros_like(ids)
    return bboxes1, ids, labels

def calculate_cent_corner_pst(img1, result1):
    cent_allclass = []
    corner_allclass = []
    center_pst = np.array([])
    corner_pst = np.array([])
    for dots in result1:
        x1 = dots[1]
        y1 = dots[2]
        x2 = dots[3]
        y2 = dots[4]
        centx = (x1 + x2) / 2
        centy = (y1 + y2) / 2
        if center_pst.size == 0:
            center_pst = np.array([[centx, centy]])
        else:
            center_pst = np.append(center_pst, [[centx, centy]], axis=0)
        if corner_pst.size == 0:
            corner_pst = np.array([[x1, y1],
                                   [x2, y2]])
        else:
            corner_pst = np.append(corner_pst, [[x1, y1], [x2, y2]], axis=0)
    center_pst = center_pst.reshape(-1, 2).astype(np.float32)
    corner_pst = corner_pst.reshape(-1, 2).astype(np.float32)
    cent_allclass = center_pst
    corner_allclass = corner_pst
    return cent_allclass, corner_allclass

def get_matched_ids_frame1(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2):
    matched_ids_cache = []
    pts_src = []
    pts_dst = []
    def find_nearest_neighbors(src_points, dst_points):
        nearest_neighbors = []
        for src_point in src_points:
            distances = np.linalg.norm(dst_points - src_point, axis=1)
            nearest_neighbor_index = np.argmin(distances)
            nearest_neighbors.append(nearest_neighbor_index)
        return nearest_neighbors

    nearest_neighbors_1_to_2 = find_nearest_neighbors(cent_allclass, cent_allclass2)
    nearest_neighbors_2_to_1 = find_nearest_neighbors(cent_allclass2, cent_allclass)

    for m, dots in enumerate(track_bboxes):
        trac_id = dots[0]
        for n, dots2 in enumerate(track_bboxes2):
            trac2_id = dots2[0]
            if trac_id == trac2_id:
                pts_src.append(cent_allclass[m])
                pts_dst.append(cent_allclass2[n])
                matched_ids_cache.append(trac_id)
                break

    for m, nearest in enumerate(nearest_neighbors_1_to_2):
        if nearest_neighbors_2_to_1[nearest] == m:
            trac_id_1 = track_bboxes[m][0]
            if trac_id_1 not in matched_ids_cache:
                pts_src.append(cent_allclass[m])
                pts_dst.append(cent_allclass2[nearest])
                matched_ids_cache.append(trac_id_1)

    pts_src = np.array(pts_src)
    pts_dst = np.array(pts_dst)
    matched_ids = matched_ids_cache.copy()
    return pts_src, pts_dst, matched_ids

def get_matched_ids(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2, corner_allclass, corner_allclass2,
                    A_max_id, B_max_id, coID_confirme):
    matched_ids_cache = []
    A_new_ID = []
    B_new_ID = []
    A_pts = []
    B_pts = []
    A_pts_corner = []
    B_pts_corner = []
    A_old_not_matched_ids = []
    B_old_not_matched_ids = []
    A_old_not_matched_pts = []
    B_old_not_matched_pts = []
    A_old_not_matched_pts_corner = []
    B_old_not_matched_pts_corner = []

    def find_nearest_neighbors(src_points, dst_points):
        nearest_neighbors = []
        for src_point in src_points:
            distances = np.linalg.norm(dst_points - src_point, axis=1)
            nearest_neighbor_index = np.argmin(distances)
            nearest_neighbors.append(nearest_neighbor_index)
        return nearest_neighbors

    nearest_neighbors_1_to_2 = find_nearest_neighbors(cent_allclass, cent_allclass2)
    nearest_neighbors_2_to_1 = find_nearest_neighbors(cent_allclass2, cent_allclass)

    for m, dots in enumerate(track_bboxes):
        trac_id = dots[0]
        if trac_id in coID_confirme:
            matched_ids_cache.append(trac_id)
            for n, dots2 in enumerate(track_bboxes2):
                trac2_id = dots2[0]
                if trac2_id == trac_id:
                    break
            continue

        if A_max_id < trac_id:
            if trac_id not in A_new_ID:
                A_new_ID.append(trac_id)
                A_pts.append(cent_allclass[m])
                A_pts_corner.append([corner_allclass[2*m], corner_allclass[2*m + 1]])
            continue

        flag_matched = 0
        for n, dots2 in enumerate(track_bboxes2):
            trac2_id = dots2[0]
            if B_max_id < trac2_id:
                if trac2_id not in B_new_ID:
                    B_new_ID.append(trac2_id)
                    B_pts.append(cent_allclass2[n])
                    B_pts_corner.append([corner_allclass2[2*n], corner_allclass2[2*n + 1]])
                continue
            if trac_id == trac2_id:
                matched_ids_cache.append(trac_id)
                flag_matched = 1
                break
        if flag_matched == 0 and trac_id not in A_old_not_matched_ids:
            A_old_not_matched_ids.append(trac_id)
            A_old_not_matched_pts.append(cent_allclass[m])
            A_old_not_matched_pts_corner.append([corner_allclass[2*m], corner_allclass[2*m + 1]])

    for n, dots2 in enumerate(track_bboxes2):
        trac2_id = dots2[0]
        if trac2_id not in matched_ids_cache and trac2_id not in B_new_ID:
            B_old_not_matched_ids.append(trac2_id)
            B_old_not_matched_pts.append(cent_allclass2[n])
            B_old_not_matched_pts_corner.append([corner_allclass2[2*n], corner_allclass2[2*n + 1]])

    for m, nearest in enumerate(nearest_neighbors_1_to_2):
        if nearest_neighbors_2_to_1[nearest] == m:
            trac_id_1 = track_bboxes[m][0]
            if trac_id_1 not in matched_ids_cache:
                matched_ids_cache.append(trac_id_1)

    matched_ids = matched_ids_cache.copy()
    return matched_ids, np.array([]), np.array([]), A_new_ID, A_pts, A_pts_corner, B_new_ID, B_pts, B_pts_corner, \
        A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, B_old_not_matched_ids, \
        B_old_not_matched_pts, B_old_not_matched_pts_corner

def A_same_target_refresh_same_ID(A_new_ID_func, A_pts_func, A_pts_corner_func, f1_func, cent_allclass2_func,
                                  track_bboxes_func, track_bboxes2_func, matched_ids_func, det_bboxes, det_bboxes2,
                                  image2, coID_confirme, thres=100):
    flag = 0
    if len(A_pts_func) == 0:
        pass
    else:
        A_pts_func = np.array(A_pts_func).reshape(-1, 1, 2).astype(np.float32)
        A_pts_corner_func = np.array(A_pts_corner_func).reshape(-1, 1, 2).astype(np.float32)
        A_dst_func = cv2.perspectiveTransform(A_pts_func, f1_func)
        A_dst_corner_func = cv2.perspectiveTransform(A_pts_corner_func, f1_func)
        if A_dst_func is not None:
            dist = np.zeros((len(A_dst_func), len(track_bboxes2_func)))
            for ii, xy in enumerate(A_dst_func):
                min_x = min(A_dst_corner_func[ii*2,0,0], A_dst_corner_func[ii*2+1,0,0])
                max_x = max(A_dst_corner_func[ii*2,0,0], A_dst_corner_func[ii*2+1,0,0])
                min_y = min(A_dst_corner_func[ii*2,0,1], A_dst_corner_func[ii*2+1,0,1])
                max_y = max(A_dst_corner_func[ii*2,0,1], A_dst_corner_func[ii*2+1,0,1])
                if min_x>1920 or max_x<0 or min_y>1080 or max_y<0:
                    continue
                else:
                    for j, dots in enumerate(cent_allclass2_func):
                        centx = int(dots[0])
                        centy = int(dots[1])
                        dist[ii,j] = ((xy[0,0]-centx)**2 + (xy[0,1]-centy)**2)**0.5
                    if len(dist[ii])>0 and min(dist[ii])<thres:
                        A_index = np.where(track_bboxes_func[:,0]==A_new_ID_func[ii])[0]
                        B_index = np.where(dist[ii]==min(dist[ii]))[0]
                        if len(A_index)>0 and len(B_index)>0:
                            A_index = A_index[0]
                            B_index = B_index[0]
                            if track_bboxes2_func[B_index,0] not in matched_ids_func:
                                min_id = int(min(track_bboxes2_func[B_index,0], A_new_ID_func[ii]))
                                track_bboxes_func[A_index,0] = min_id
                                track_bboxes2_func[B_index,0] = min_id
                                coID_confirme.append(int(min_id))
                                matched_ids_func.append(int(min_id))
    return track_bboxes_func, track_bboxes2_func, matched_ids_func, flag, coID_confirme

def B_same_target_refresh_same_ID(B_new_ID, B_pts, B_pts_corner, f2, cent_allclass,
                                  track_bboxes, track_bboxes2, matched_ids, det_bboxes, det_bboxes2,
                                  image2, coID_confirme, thres=150):
    flag = 0
    if len(B_pts) == 0:
        pass
    else:
        B_pts = np.array(B_pts).reshape(-1,1,2).astype(np.float32)
        B_pts_corner = np.array(B_pts_corner).reshape(-1,1,2).astype(np.float32)
        B_dst = cv2.perspectiveTransform(B_pts, f2)
        B_dst_corner = cv2.perspectiveTransform(B_pts_corner, f2)
        if B_pts is not None:
            dist = np.zeros((len(B_dst), len(track_bboxes)))
            for ii, xy in enumerate(B_dst):
                min_x = min(B_dst_corner[ii*2,0,0], B_dst_corner[ii*2+1,0,0])
                max_x = max(B_dst_corner[ii*2,0,0], B_dst_corner[ii*2+1,0,0])
                min_y = min(B_dst_corner[ii*2,0,1], B_dst_corner[ii*2+1,0,1])
                max_y = max(B_dst_corner[ii*2,0,1], B_dst_corner[ii*2+1,0,1])
                if min_x>1920 or max_x<0 or min_y>1080 or max_y<0:
                    continue
                else:
                    for j,dots in enumerate(cent_allclass):
                        centx = int(dots[0])
                        centy = int(dots[1])
                        dist[ii,j] = ((xy[0,0]-centx)**2+(xy[0,1]-centy)**2)**0.5
                    if len(dist[ii])>0 and min(dist[ii])<thres:
                        B_index = np.where(track_bboxes2[:,0]==B_new_ID[ii])[0]
                        A_index = np.where(dist[ii]==min(dist[ii]))[0]
                        if len(A_index)>0 and len(B_index)>0:
                            A_index = A_index[0]
                            B_index = B_index[0]
                            if track_bboxes[A_index,0] not in matched_ids:
                                min_id = min(track_bboxes[A_index,0], B_new_ID[ii])
                                track_bboxes[A_index,0] = min_id
                                track_bboxes2[B_index,0] = min_id
                                coID_confirme.append(int(min_id))
                                matched_ids.append(int(min_id))
    return track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme

def same_target_refresh_same_ID(A_new_ID_func, A_pts_func, A_pts_corner_func, f1_func, cent_allclass2_func,
                                track_bboxes_func, track_bboxes2_func, matched_ids_func, det_bboxes, det_bboxes2,
                                image2, coID_confirme, thres=100):
    flag = 0
    if len(A_pts_func) == 0:
        pass
    else:
        A_pts_func = np.array(A_pts_func).reshape(-1,1,2).astype(np.float32)
        A_pts_corner_func = np.array(A_pts_corner_func).reshape(-1,1,2).astype(np.float32)
        A_dst_func = cv2.perspectiveTransform(A_pts_func, f1_func)
        A_dst_corner_func = cv2.perspectiveTransform(A_pts_corner_func, f1_func)
        if A_dst_func is not None:
            dist = np.zeros((len(A_dst_func), len(track_bboxes2_func)))
            for ii, xy in enumerate(A_dst_func):
                min_x = min(A_dst_corner_func[ii*2,0,0], A_dst_corner_func[ii*2+1,0,0])
                max_x = max(A_dst_corner_func[ii*2,0,0], A_dst_corner_func[ii*2+1,0,0])
                min_y = min(A_dst_corner_func[ii*2,0,1], A_dst_corner_func[ii*2+1,0,1])
                max_y = max(A_dst_corner_func[ii*2,0,1], A_dst_corner_func[ii*2+1,0,1])
                if min_x>1920 or max_x<0 or min_y>1080 or max_y<0:
                    continue
                else:
                    for j,dots in enumerate(cent_allclass2_func):
                        centx = int(dots[0])
                        centy = int(dots[1])
                        dist[ii,j] = ((xy[0,0]-centx)**2+(xy[0,1]-centy)**2)**0.5
                    if len(dist[ii])>0 and min(dist[ii])<thres:
                        A_index = np.where(track_bboxes_func[:,0]==A_new_ID_func[ii])[0]
                        B_index = np.where(dist[ii]==min(dist[ii]))[0]
                        if len(A_index)>0 and len(B_index)>0:
                            A_index = A_index[0]
                            B_index = B_index[0]
                            if track_bboxes2_func[B_index,0] not in matched_ids_func:
                                min_id = int(min(track_bboxes2_func[B_index,0], A_new_ID_func[ii]))
                                track_bboxes_func[A_index,0] = min_id
                                track_bboxes2_func[B_index,0] = min_id
                                coID_confirme.append(int(min_id))
                                matched_ids_func.append(int(min_id))
    return track_bboxes_func, track_bboxes2_func, matched_ids_func, flag, coID_confirme

def cross_view_id_matching(track_bboxes, track_bboxes2, reid_feats_A, reid_feats_B, lambda_pos=0.5):
    if len(track_bboxes)==0 or len(track_bboxes2)==0:
        return track_bboxes, track_bboxes2

    ids_A = track_bboxes[:,0].astype(int)
    ids_B = track_bboxes2[:,0].astype(int)

    cent_A = np.array([((bb[1]+bb[3])/2, (bb[2]+bb[4])/2) for bb in track_bboxes])
    cent_B = np.array([((bb[1]+bb[3])/2, (bb[2]+bb[4])/2) for bb in track_bboxes2])

    pos_dist = np.zeros((len(track_bboxes), len(track_bboxes2)))
    for i in range(len(track_bboxes)):
        for j in range(len(track_bboxes2)):
            pos_dist[i,j] = np.linalg.norm(cent_A[i]-cent_B[j])

    reid_dist = np.zeros((len(track_bboxes), len(track_bboxes2)))
    for i in range(len(track_bboxes)):
        for j in range(len(track_bboxes2)):
            fA = reid_feats_A[i]
            fB = reid_feats_B[j]
            cos_sim = np.dot(fA,fB)/(np.linalg.norm(fA)*np.linalg.norm(fB)+1e-6)
            reid_dist[i,j] = 1-cos_sim

    cost_matrix = lambda_pos*pos_dist + (1-lambda_pos)*reid_dist
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for r,c in zip(row_ind,col_ind):
        unified_id = min(ids_A[r], ids_B[c])
        track_bboxes[r,0] = unified_id
        track_bboxes2[c,0] = unified_id

    return track_bboxes, track_bboxes2


##########################################
# 5. 主函数
##########################################
def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='配置文件路径',
        default='D:\\Multi-Drone-Multi-Object-Detection-and-Tracking-main\\demo\\configs\\mot/bytetrack\\swin_transformer_bytetrack_Adam.py')
    parser.add_argument('--input', help='输入视频文件或文件夹',
        default='D:\\Multi-Drone-Multi-Object-Detection-and-Tracking-main\\data\\MDMT/test/1/')
    parser.add_argument('--xml_dir', help='真值 XML 文件目录',
        default='D:\\Multi-Drone-Multi-Object-Detection-and-Tracking-main\\data\\MDMT/new_xml/1/')
    parser.add_argument('--result_dir', help='结果保存目录', default='./result/multiDrone_BTORBM-LSTM_TSIN')
    parser.add_argument('--method', help='方法名称', default='Transformer_Integration')
    parser.add_argument('--output', help='输出视频文件或文件夹', default='./workdirs/viewA.mp4')
    parser.add_argument('--output2', help='输出视频文件或文件夹', default='./workdirs/viewB.mp4')
    parser.add_argument('--checkpoint', help='检测/跟踪模型检查点文件路径')
    parser.add_argument('--lstm-checkpoint', help='LSTM轨迹预测模型权重文件', default='D:\\Multi-Drone-Multi-Object-Detection-and-Tracking-main\\data\\MDMT\\csv\\checkpoint_epoch30_lstm.pth')
    parser.add_argument('--score-thr', type=float, default=0.0, help='过滤边界框的分数阈值')
    parser.add_argument('--device', default='cuda:0', help='使用的设备')
    parser.add_argument('--show', action='store_true', help='是否实时显示结果')
    parser.add_argument('--backend', choices=['cv2', 'plt'], default='cv2', help='可视化后端')
    parser.add_argument('--fps', type=int, default=10, help='输出视频的 FPS')
    args = parser.parse_args()
    assert args.output or args.show

    from mmtrack.apis import inference_mot, init_model

    device = torch.device(args.device)

    # 1) 初始化检测和跟踪模型
    model = init_model(args.config, args.checkpoint, device=device)
    model2 = init_model(args.config, args.checkpoint, device=device)

    # 2) 初始化 ReID 模型
    checkpoint_path = 'D:\\Multi-Drone-Multi-Object-Detection-and-Tracking-main\\demo\\checkpoint\\transreid\\deit_transreid_veri.pth'
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    reid_model = MyReIDModel(feature_dim=768)
    reid_model.load_state_dict(ckpt, strict=False)
    reid_model.eval().to(device)

    # 3) 初始化 ORB 与Transformer特征增强模块
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    transformer = FeatureEnhancementTransformer(embed_dim=64, num_heads=8, num_layers=4).to(device)

    # 4) 初始化 LSTM 轨迹预测模型
    trajectory_predictor = LSTMTrajectoryPredictor(
        input_dim=2, hidden_dim=32, num_layers=1, seq_len=5, pred_len=3
    ).to(device)
    if args.lstm_checkpoint and osp.exists(args.lstm_checkpoint):
        print(f"Loading LSTM checkpoint from {args.lstm_checkpoint}")
        lstm_ckpt = torch.load(args.lstm_checkpoint, map_location='cpu')
        trajectory_predictor.load_state_dict(lstm_ckpt, strict=False)
    else:
        print("No LSTM checkpoint provided. Using random initialization (可能预测不准确)!")

    trajectory_predictor.eval()

    # 两个字典记录视角 A 和 B 的历史轨迹
    traj_history_A = {}
    traj_history_B = {}

    # 上一帧的跟踪结果
    track_bboxes_old = np.empty((0,6))
    track_bboxes2_old = np.empty((0,6))

    time_start_all = time.time()

    for dirrr in sorted(os.listdir(args.input)):
        if "-2" in dirrr:
            print("dirrr has -2")
            continue
        if "-1" not in dirrr and "-2" not in dirrr:
            continue

        sequence_dir = os.path.join(args.input + dirrr + "/")
        if osp.isdir(sequence_dir):
            imgs = sorted(
                filter(lambda x: x.endswith(('.jpg','.png','.jpeg')),
                       os.listdir(sequence_dir)),
                key=lambda x: int(x.split('.')[0]))
            IN_VIDEO = False
        else:
            imgs = mmcv.VideoReader(sequence_dir)
            IN_VIDEO = True

        if args.output is not None:
            if args.output.endswith('.mp4'):
                OUT_VIDEO = True
                out_dir = tempfile.TemporaryDirectory()
                out_path = out_dir.name
                _out = args.output.rsplit(os.sep,1)
                if len(_out)>1:
                    os.makedirs(_out[0],exist_ok=True)
            else:
                OUT_VIDEO = False
                out_path = args.output
                os.makedirs(out_path, exist_ok=True)
        if args.output2 is not None:
            if args.output2.endswith('.mp4'):
                OUT_VIDEO = True
                out_dir2 = tempfile.TemporaryDirectory()
                out_path2 = out_dir2.name
                _out2 = args.output2.rsplit(os.sep,1)
                if len(_out2)>1:
                    os.makedirs(_out2[0],exist_ok=True)
            else:
                OUT_VIDEO = False
                out_path2 = args.output2
                os.makedirs(out_path2, exist_ok=True)

        fps = args.fps
        if args.show or OUT_VIDEO:
            if fps is None and IN_VIDEO:
                fps = imgs.fps
            if not fps:
                raise ValueError('Please set the FPS for the output video.')
            fps = int(fps)

        prog_bar = mmcv.ProgressBar(len(imgs))

        matched_ids = []
        A_max_id = 0
        B_max_id = 0
        result_dict = {}
        result_dict2 = {}
        coID_confirme = []
        time_start = time.time()

        f1_last, f2_last = None, None

        for i, img in enumerate(imgs):
            if isinstance(img,str):
                # 读取图像路径
                img = osp.join(sequence_dir, img)
                img2 = img.replace("/1/", "/2/").replace("-1","-2")
                image1 = cv2.imread(img)
                image2 = cv2.imread(img2)

                keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
                keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
                if descriptors1 is None or descriptors2 is None:
                    if i==0:
                        sequence1 = img.split("/")[-2]
                        sequence2 = img2.split("/")[-2]
                    prog_bar.update()
                    continue
                descriptors1 = torch.tensor(descriptors1,dtype=torch.float32).to(device).unsqueeze(1)
                descriptors2 = torch.tensor(descriptors2,dtype=torch.float32).to(device).unsqueeze(1)
                with torch.no_grad():
                    ed1 = transformer(descriptors1).squeeze(1)
                    ed2 = transformer(descriptors2).squeeze(1)
                ed1 = ed1.cpu().numpy().astype(np.float32)
                ed2 = ed2.cpu().numpy().astype(np.float32)

                try:
                    matches = bf.match(np.uint8(ed1),np.uint8(ed2))
                    matches = sorted(matches, key=lambda x:x.distance)
                except cv2.error as e:
                    print(f"Frame {i}: Matching error - {e}")
                    prog_bar.update()
                    continue
            else:
                # 视频的情况
                continue

            if i==0:
                # 第0帧
                sequence1 = img.split("/")[-2]
                xml_file1 = os.path.join(args.xml_dir + f"{sequence1}.xml")
                sequence2 = img2.split("/")[-2]
                xml_file2 = os.path.join(args.xml_dir + f"{sequence2}.xml")

                bboxes1, ids1, labels1 = read_xml_r(xml_file1, i)
                bboxes2, ids2, labels2 = read_xml_r(xml_file2, i)
                max_id = max(A_max_id, B_max_id)

                result, max_id = inference_mot(model, img, frame_id=i, bboxes1=bboxes1, ids1=ids1, labels1=labels1, max_id=max_id)
                det_bboxes = result['det_bboxes'][0]
                track_bboxes = result['track_bboxes'][0]

                result2, max_id = inference_mot(model2, img2, frame_id=i, bboxes1=bboxes2, ids1=ids2, labels1=labels2, max_id=max_id)
                det_bboxes2 = result2['det_bboxes'][0]
                track_bboxes2 = result2['track_bboxes'][0]

                result_dict[f"frame={i}"] = track_bboxes[:,0:5].tolist()
                result_dict2[f"frame={i}"] = track_bboxes2[:,0:5].tolist()

                cent_allclass, corner_allclass = calculate_cent_corner_pst(image1, track_bboxes)
                cent_allclass2, corner_allclass2 = calculate_cent_corner_pst(image2, track_bboxes2)

                if len(matches) >=4:
                    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,2)
                    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1,2)
                    f1, f1_last = compute_transf_matrix(src_pts, dst_pts, f1_last, image1,image2)
                    f2, f2_last = compute_transf_matrix(dst_pts, src_pts, f2_last, image2,image1)
                else:
                    f1,f2 = None, None

                pts_src, pts_dst, matched_ids = get_matched_ids_frame1(track_bboxes, track_bboxes2, cent_allclass, cent_allclass2)
                A_max_id = max(A_max_id, max(track_bboxes[:,0]) if len(track_bboxes)>0 else 0)
                B_max_id = max(B_max_id, max(track_bboxes2[:,0]) if len(track_bboxes2)>0 else 0)

                if args.output is not None:
                    if IN_VIDEO or OUT_VIDEO:
                        out_file = osp.join(out_path, f'{i:06d}.jpg')
                        out_file2 = osp.join(out_path2, f'{i:06d}.jpg')
                    else:
                        out_file = osp.join(out_path, img.rsplit(os.sep,1)[-1])
                        out_file2 = osp.join(out_path2, img.rsplit(os.sep,1)[-1])
                else:
                    out_file = None

                model.show_result(
                    img, result, score_thr=args.score_thr, show=args.show,
                    wait_time=int(1000./fps) if fps else 0,
                    out_file=out_file, backend=args.backend
                )
                model2.show_result(
                    img2, result2, score_thr=args.score_thr, show=args.show,
                    wait_time=int(1000./fps) if fps else 0,
                    out_file=out_file2, backend=args.backend
                )

                # 计算对这帧的误差(此时还没有未来帧的预测)
                compute_and_store_error_for_frame(i, track_bboxes)
                compute_and_store_error_for_frame(i, track_bboxes2)

                # 在检测结果可视化图像上叠加轨迹预测
                if out_file and osp.exists(out_file):
                    vis_img1 = cv2.imread(out_file)
                    if vis_img1 is not None:
                        update_and_predict(traj_history_A, track_bboxes, vis_img1, trajectory_predictor,
                                           color=(0,255,0), frame_idx=i)
                        cv2.imwrite(out_file, vis_img1)
                if out_file2 and osp.exists(out_file2):
                    vis_img2 = cv2.imread(out_file2)
                    if vis_img2 is not None:
                        update_and_predict(traj_history_B, track_bboxes2, vis_img2, trajectory_predictor,
                                           color=(0,0,255), frame_idx=i)
                        cv2.imwrite(out_file2, vis_img2)

                prog_bar.update()
                track_bboxes_old = track_bboxes.copy()
                track_bboxes2_old = track_bboxes2.copy()
                continue
            else:
                # 后续帧
                if len(track_bboxes_old)>0:
                    bboxes1 = torch.tensor(track_bboxes_old[:,1:5], dtype=torch.long)
                    ids1 = torch.tensor(track_bboxes_old[:,0], dtype=torch.long)
                    labels1 = torch.zeros_like(ids1)
                else:
                    bboxes1 = torch.empty((0,4), dtype=torch.long)
                    ids1 = torch.empty((0,), dtype=torch.long)
                    labels1 = torch.empty((0,), dtype=torch.long)

                if len(track_bboxes2_old)>0:
                    bboxes2 = torch.tensor(track_bboxes2_old[:,1:5], dtype=torch.long)
                    ids2 = torch.tensor(track_bboxes2_old[:,0], dtype=torch.long)
                    labels2 = torch.zeros_like(ids2)
                else:
                    bboxes2 = torch.empty((0,4), dtype=torch.long)
                    ids2 = torch.empty((0,), dtype=torch.long)
                    labels2 = torch.empty((0,), dtype=torch.long)

            if len(matches)>=4:
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1,2)
                f1, f1_last = compute_transf_matrix(src_pts, dst_pts, f1_last, image1, image2)
                f2, f2_last = compute_transf_matrix(dst_pts, src_pts, f2_last, image2, image1)
            else:
                f1,f2 = f1_last, f2_last

            max_id = max(A_max_id,B_max_id)
            result, max_id = inference_mot(model, img, frame_id=i,
                bboxes1=bboxes1, ids1=ids1, labels1=labels1, max_id=max_id)
            det_bboxes = result['det_bboxes'][0]
            track_bboxes = result['track_bboxes'][0]

            result2, max_id = inference_mot(model2, img2, frame_id=i,
                bboxes1=bboxes2, ids1=ids2, labels1=labels2, max_id=max_id)
            det_bboxes2 = result2['det_bboxes'][0]
            track_bboxes2 = result2['track_bboxes'][0]

            result_dict[f"frame={i}"] = track_bboxes[:,0:5].tolist() if len(track_bboxes)>0 else []
            result_dict2[f"frame={i}"] = track_bboxes2[:,0:5].tolist() if len(track_bboxes2)>0 else []

            cent_allclass, corner_allclass = calculate_cent_corner_pst(image1, track_bboxes)
            cent_allclass2, corner_allclass2 = calculate_cent_corner_pst(image2, track_bboxes2)

            matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, \
            B_new_ID, B_pts, B_pts_corner, A_old_not_matched_ids, A_old_not_matched_pts, \
            A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, \
            B_old_not_matched_pts_corner = get_matched_ids(
                track_bboxes, track_bboxes2, cent_allclass, cent_allclass2,
                corner_allclass, corner_allclass2, A_max_id, B_max_id, coID_confirme
            )

            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = A_same_target_refresh_same_ID(
                A_new_ID, A_pts, A_pts_corner, f1, cent_allclass2, track_bboxes, track_bboxes2, matched_ids,
                det_bboxes, det_bboxes2, image2, coID_confirme, thres=80
            )

            matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, \
            B_new_ID, B_pts, B_pts_corner, A_old_not_matched_ids, A_old_not_matched_pts, \
            A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, \
            B_old_not_matched_pts_corner = get_matched_ids(
                track_bboxes, track_bboxes2, cent_allclass, cent_allclass2,
                corner_allclass, corner_allclass2, A_max_id, B_max_id, coID_confirme
            )

            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = B_same_target_refresh_same_ID(
                B_new_ID, B_pts, B_pts_corner, f2, cent_allclass, track_bboxes, track_bboxes2,
                matched_ids, det_bboxes, det_bboxes2, image1, coID_confirme, thres=80
            )

            matched_ids, pts_src, pts_dst, A_new_ID, A_pts, A_pts_corner, \
            B_new_ID, B_pts, B_pts_corner, A_old_not_matched_ids, A_old_not_matched_pts, \
            A_old_not_matched_pts_corner, B_old_not_matched_ids, B_old_not_matched_pts, \
            B_old_not_matched_pts_corner = get_matched_ids(
                track_bboxes, track_bboxes2, cent_allclass, cent_allclass2,
                corner_allclass, corner_allclass2, A_max_id, B_max_id, coID_confirme
            )

            track_bboxes, track_bboxes2, matched_ids, flag, coID_confirme = same_target_refresh_same_ID(
                A_old_not_matched_ids, A_old_not_matched_pts, A_old_not_matched_pts_corner, f1,
                cent_allclass2, track_bboxes, track_bboxes2, matched_ids,
                det_bboxes, det_bboxes2, image2, coID_confirme, thres=50
            )

            if len(track_bboxes)>0:
                A_max_id = max(A_max_id, max(track_bboxes[:,0]))
            if len(track_bboxes2)>0:
                B_max_id = max(B_max_id, max(track_bboxes2[:,0]))

            # 提取并做ReID
            reid_feats_A = extract_reid_features(reid_model, image1, track_bboxes)
            reid_feats_B = extract_reid_features(reid_model, image2, track_bboxes2)
            track_bboxes, track_bboxes2 = cross_view_id_matching(
                track_bboxes, track_bboxes2, reid_feats_A, reid_feats_B, lambda_pos=0.5
            )

            # NMS
            thresh=0.2
            if len(track_bboxes)>0:
                track_bboxes = all_nms(track_bboxes, thresh)
            if len(track_bboxes2)>0:
                track_bboxes2 = all_nms(track_bboxes2, thresh)

            result['track_bboxes'][0] = track_bboxes
            result2['track_bboxes'][0] = track_bboxes2
            track_bboxes_old = track_bboxes.copy()
            track_bboxes2_old = track_bboxes2.copy()

            result_dict[f"frame={i}"] = track_bboxes[:,0:5].tolist() if len(track_bboxes)>0 else []
            result_dict2[f"frame={i}"] = track_bboxes2[:,0:5].tolist() if len(track_bboxes2)>0 else []

            if args.output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{i:06d}.jpg')
                    out_file2 = osp.join(out_path2, f'{i:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep,1)[-1])
                    out_file2 = osp.join(out_path2, img.rsplit(os.sep,1)[-1])
            else:
                out_file=None

            model.show_result(
                img, result, score_thr=args.score_thr, show=args.show,
                wait_time=int(1000./fps) if fps else 0, out_file=out_file, backend=args.backend
            )
            model2.show_result(
                img2, result2, score_thr=args.score_thr, show=args.show,
                wait_time=int(1000./fps) if fps else 0, out_file=out_file2, backend=args.backend
            )

            # 每帧结束后先对本帧的真实目标中心与之前的预测做一次误差计算
            compute_and_store_error_for_frame(i, track_bboxes)
            compute_and_store_error_for_frame(i, track_bboxes2)

            # 轨迹预测
            if out_file and osp.exists(out_file):
                vis_img1 = cv2.imread(out_file)
                if vis_img1 is not None:
                    update_and_predict(traj_history_A, track_bboxes, vis_img1, trajectory_predictor,
                                       color=(0,255,0), frame_idx=i)
                    cv2.imwrite(out_file, vis_img1)
            if out_file2 and osp.exists(out_file2):
                vis_img2 = cv2.imread(out_file2)
                if vis_img2 is not None:
                    update_and_predict(traj_history_B, track_bboxes2, vis_img2, trajectory_predictor,
                                       color=(0,0,255), frame_idx=i)
                    cv2.imwrite(out_file2, vis_img2)

            prog_bar.update()

        # 单个序列结束
        time_end = time.time()
        method = args.method
        json_dir = f"{args.result_dir}/{method}/"
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        with open(f"{json_dir}/{sequence1}.json","w") as f:
            json.dump(result_dict, f, indent=4)
            print("第一个视角输出文件写入完成！")
        with open(f"{json_dir}/{sequence2}.json","w") as f2:
            json.dump(result_dict2, f2, indent=4)
            print("第二个视角输出文件写入完成！")
        with open(f"{json_dir}/time.txt","a") as f3:
            f3.write(f"{sequence1} time consume :{time_end-time_start}\n")
            print("输出文件time.txt写入完成！")

        if args.output and OUT_VIDEO:
            print(f'making the output video at {args.output} with a FPS of {fps}')
            mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
            mmcv.frames2video(out_path2, args.output2, fps=fps, fourcc='mp4v')
            out_dir.cleanup()

    # 所有序列结束，做整体时间记录
    time_end_all = time.time()
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    with open(f"{json_dir}/time.txt","a") as f3:
        f3.write(f"ALL time consume :{time_end_all-time_start_all}\n")

    save_dir = "D:\\Multi-Drone-Multi-Object-Detection-and-Tracking-main\\demo\\prediction_picture"
    plot_and_save_all_errors(save_dir)
    print(f"所有误差可视化图已保存到 {save_dir}")

if __name__ == '__main__':
    main()
