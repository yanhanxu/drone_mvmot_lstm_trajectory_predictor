import os
import xml.etree.ElementTree as ET
import csv
import re

def parse_xml_to_csv(xml_dir, output_csv, img_width=1920, img_height=1080):
    """
    在xml_dir目录下找所有*.xml文件，
    每个文件形如  1-1.xml 或 2-2.xml
    解析后输出 CSV，列包含：
       seq_id, vantage, obj_id, frame_idx, x_center, y_center
    可以后续再做归一化 (x/img_width, y/img_height).
    """
    # 打开csv写文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow(["seq_id", "vantage", "obj_id", "frame_idx", "x_center", "y_center"])

        # 遍历xml_dir下所有xml
        for filename in sorted(os.listdir(xml_dir)):
            if not filename.endswith(".xml"):
                continue

            # 假设文件名类似 "22-1.xml" or "2-2.xml"
            # 解析出 seq_id, vantage
            # 你可以自定义怎么解析，也可用正则
            # 例如： 22-1 => seq_id=22, vantage=1
            base = os.path.splitext(filename)[0]  # "22-1"
            # 用个简单正则
            m = re.match(r"(\d+)-(\d+)", base)
            if not m:
                # 如果你的文件命名不是这种格式，需要你自己处理
                seq_id = base
                vantage = "unknown"
            else:
                seq_id = m.group(1)
                vantage = m.group(2)

            xml_path = os.path.join(xml_dir, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 找所有<track>标签
            tracks = root.findall('track')
            for track in tracks:
                obj_id = track.attrib['id']
                boxes = track.findall('box')

                for box in boxes:
                    frame_idx = int(box.attrib['frame'])
                    outside = int(box.attrib.get('outside','0'))
                    # 如果 outside=1 说明目标已离开画面，可以选择跳过
                    # 如果要保留离场轨迹，也可不跳过
                    if outside == 1:
                        continue

                    xtl = float(box.attrib['xtl'])
                    ytl = float(box.attrib['ytl'])
                    xbr = float(box.attrib['xbr'])
                    ybr = float(box.attrib['ybr'])

                    x_center = (xtl + xbr)/2
                    y_center = (ytl + ybr)/2

                    # 你也可以直接在此处归一化
                    # x_center /= img_width
                    # y_center /= img_height

                    writer.writerow([seq_id, vantage, obj_id, frame_idx, x_center, y_center])

    print(f"Done. CSV saved at {output_csv}")


if __name__ == "__main__":
    xml_dir = r"D:\Multi-Drone-Multi-Object-Detection-and-Tracking-main\data\MDMT\csv\all_xml"
    output_csv = r"D:\Multi-Drone-Multi-Object-Detection-and-Tracking-main\data\MDMT\csv\all_csv_output.csv"
    parse_xml_to_csv(xml_dir, output_csv)
