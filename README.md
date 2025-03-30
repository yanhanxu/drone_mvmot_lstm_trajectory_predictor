📆 Project Highlights

✅ Multi-view object detection and tracking

♻️ Cross-view ID matching and merging

🧠 Transformer-based feature enhancement

👤 ReID-based feature fusion for identity association

🔮 LSTM-based trajectory prediction (ADE/FDE metrics)

📊 Visualization of prediction errors (histogram, curves, ADE/FDE examples)

📁 Project Structure

drone_mvmot_lstm_trajectory_predictor/
├── demo/                                # Main scripts and visualizations
│   ├── your_main_script.py              # Main execution script
│   ├── prediction_picture/              # Visualized prediction errors
│   └── configs/                         # mmtrack config files
│
├── data/                                # Dataset
│   └── MDMT/                            # Multi-drone multi-target dataset
│       ├── test/                        # Image sequences
│       └── new_xml/                     # Ground-truth trajectory (CVAT XML)
│
├── checkpoint/                          # Model weights
│   └── transreid/
│       └── deit_transreid.pth
│
├── result/                              # Tracking results and logs
├── requirements.txt                     # Python dependencies
└── README.md                            # This file

🛠️ Environment Setup

Recommended: Python 3.8, PyTorch 1.7+, Anaconda preferred.

Install Dependencies

pip install -r requirements.txt

Sample requirements.txt:

torch>=1.7.0
torchvision
mmcv-full==1.5.0
mmdet==2.25.1
mmtrack==0.12.0
opencv-python
numpy
matplotlib
scipy
Pillow

🚀 Quick Start

1. Prepare Dataset

Place your image sequences and annotation files:

data/MDMT/test/       # Sequence images (e.g., 1-1, 1-2)
data/MDMT/new_xml/    # Ground truth XMLs

2. Prepare Models

Download:

LSTM model checkpoint (e.g., checkpoint_epoch30_lstm.pth)

ReID model (e.g., deit_transreid.pth)

Place them in the checkpoint/ directory and update the script paths accordingly.

3. Run the Pipeline

python demo/your_main_script.py \
  --config demo/configs/mot/bytetrack/swin_transformer_bytetrack_Adam.py \
  --input data/MDMT/test/1/ \
  --xml_dir data/MDMT/new_xml/1/ \
  --lstm-checkpoint ./checkpoint/lstm/checkpoint_epoch30_lstm.pth \
  --output ./result/outputA.mp4 \
  --output2 ./result/outputB.mp4 \
  --device cuda:0

📈 Visual Outputs

Prediction results and error analysis will be saved to:

demo/prediction_picture/

Includes:

Average / Max / Median prediction errors

ADE / FDE visual examples

Histogram of all prediction errors

Error trend over frames

(Optional) Per-object error curves

📊 Metrics Explained

Metric

Description

ADE

Average Displacement Error (average distance over all predicted steps)

FDE

Final Displacement Error (distance of final predicted step)

Max Error

Maximum point-wise prediction error

Error Histogram

Distribution of prediction errors

📆 Example Args

python your_main_script.py --show --fps 10 --backend cv2

📄 Recommended .gitignore

*.pth
*.mp4
*.avi
*.npy
*.json
*.log
*.csv
result/
demo/prediction_picture/

📓 License

This project is licensed under the MIT License. Model weights are for research only.

🙌 Acknowledgements

Built upon and inspired by:

mmtracking

ByteTrack

TransReID

For questions, issues or pull requests, feel free to contribute
