"""
YOLOv8n 迁移学习训练脚本
数据集：仿真生成的探地雷达B-scan图像，目标为地下PVC管道
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO, RTDETR
# Windows 多进程 DLL 加载修复
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ======================== 模型配置 ========================
MODEL_TYPE = "rtdetr"      # "yolo" 或 "rtdetr"

YOLO_WEIGHT = "yolov8n.pt"
RTDETR_WEIGHT = "rtdetr-l.pt"

# ======================== 配置 ========================
ROOT = Path(__file__).resolve().parent
FIGS_DIR = ROOT / "dataset_all" / "train" / "images"
LABELS_DIR = ROOT / "dataset_all" / "train" / "labels"
DATASET_DIR = ROOT / "dataset_all"
data_yaml = DATASET_DIR / "data.yaml"


# 训练超参数
EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 640  # 0 表示使用原始分辨率，不缩放
DEVICE = "cuda"  # GPU编号，无GPU时改为 "cpu"

# ======================== 训练 ========================
def train(data_yaml):

    if MODEL_TYPE.lower() == "yolo":
        model = YOLO(YOLO_WEIGHT)
        run_name = "yolov8n_big"
    elif MODEL_TYPE.lower() == "rtdetr":
        model = RTDETR(RTDETR_WEIGHT)
        run_name = "rtdetr_l"
    else:
        raise ValueError(
            f"Unsupported MODEL_TYPE: {MODEL_TYPE}"
        )

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        workers=0,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=str(ROOT / "runs"),
        name=run_name,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.01,
        warmup_epochs=5,
        augment=True,
        fliplr=0.5,
        patience=20,
        amp=True
    )

    print(
        f"训练完成！最佳权重保存在 "
        f"runs/{run_name}/weights/best.pt"
    )

    return model


# ======================== 主流程 ========================
if __name__ == "__main__":
    train(data_yaml)


