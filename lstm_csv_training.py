import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------------
# 1. LSTM 轨迹预测模型
# -------------------------------
class LSTMTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=3, seq_len=10, pred_len=6):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim * pred_len)

    def forward(self, traj):
        """
        traj: [batch_size, seq_len, 2]
        return: [batch_size, pred_len, 2]
        """
        bsz = traj.size(0)
        x = self.input_embedding(traj)       # -> [bsz, seq_len, hidden_dim]
        x, _ = self.lstm(x)                  # -> [bsz, seq_len, hidden_dim]
        x = x[:, -1]                         # -> [bsz, hidden_dim]
        out = self.head(x)                   # -> [bsz, pred_len*2]
        return out.view(bsz, self.pred_len, 2)

# -------------------------------
# 2. 轨迹数据集定义
# -------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, seq_len=10, pred_len=6):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        df = pd.read_csv(csv_file)
        df = df.sort_values(by=["seq_id", "vantage", "obj_id", "frame_idx"])

        self.samples = []
        grouped = df.groupby(["seq_id", "vantage", "obj_id"])
        for (seq, van, oid), group in grouped:
            arr = group[["frame_idx", "x_center", "y_center"]].values
            for i in range(len(arr) - (seq_len + pred_len) + 1):
                clip = arr[i : i + seq_len + pred_len]
                inp_xy = clip[:seq_len, 1:3]
                lab_xy = clip[seq_len:, 1:3]
                self.samples.append((inp_xy, lab_xy))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, lab = self.samples[idx]
        inp = torch.tensor(inp, dtype=torch.float32)
        lab = torch.tensor(lab, dtype=torch.float32)
        return inp, lab

# -------------------------------
# 3. 训练函数
# -------------------------------
def train_lstm(args):
    dataset = TrajectoryDataset(
        csv_file=args.csv_file,
        seq_len=args.seq_len,
        pred_len=args.pred_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LSTMTrajectoryPredictor(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        pred_len=args.pred_len
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    if args.load_ckpt and os.path.exists(args.load_ckpt):
        print(f"Loading checkpoint from {args.load_ckpt}")
        state_dict = torch.load(args.load_ckpt, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_in, batch_lab in dataloader:
            batch_in = batch_in.to(args.device)
            batch_lab = batch_lab.to(args.device)

            pred = model(batch_in)
            loss = criterion(pred, batch_lab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_in.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.6f}")

    if args.save_ckpt:
        torch.save(model.state_dict(), args.save_ckpt)
        print(f"Model saved to {args.save_ckpt}")

# -------------------------------
# 4. 启动训练
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', required=True, default='D:\Multi-Drone-Multi-Object-Detection-and-Tracking-main\data\MDMT\csv.csv')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=6)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_ckpt', default='./checkpoint_lstm_pred6.pth')
    parser.add_argument('--load_ckpt', default=None)
    args = parser.parse_args()

    train_lstm(args)

if __name__ == "__main__":
    main()
