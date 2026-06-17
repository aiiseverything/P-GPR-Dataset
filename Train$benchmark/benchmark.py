import os
import csv
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from _baseline_utils import compute_ap
from ultralytics import YOLO, RTDETR

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn
)

from torchvision.models.detection.retinanet import (
    RetinaNetClassificationHead
)
import matplotlib.pyplot as plt

# =====================================================
# 配置
# =====================================================

ROOT = Path(__file__).resolve().parent

TEST_DIR = ROOT / "dataset_all" / "test"

YOLO_WEIGHT = ROOT / "runs" / "yolov8n_big-2" / "weights" / "best.pt"

RTDETR_WEIGHT = ROOT / "runs" / "rtdetr_l-4" / "weights" / "best.pt"

FRCNN_WEIGHT = ROOT / "runs" / "frcnn" / "best.pth"

RETINANET_WEIGHT = ROOT / "runs" / "retinanet" / "best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_THRESH = 0.25

IOU_LIST = np.arange(
    0.50,
    1.00,
    0.05
)
CLASS_NAMES = [
    "steel",
    "pipe"
]


# =====================================================
# YOLO标签读取
# =====================================================

def load_yolo_label(label_path, w, h):

    boxes = []
    labels = []

    if not os.path.exists(label_path):
        return boxes, labels

    with open(label_path, "r") as f:

        for line in f:

            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls))

    return boxes, labels


# =====================================================
# FasterRCNN
# =====================================================

def build_frcnn():

    model = fasterrcnn_resnet50_fpn(weights=None)

    in_feat = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_feat,
        3
    )

    return model

# =====================================================
# retinanet
# =====================================================


def build_retinanet():

    model = retinanet_resnet50_fpn(
        weights=None
    )

    num_anchors = (
        model.head.classification_head.num_anchors
    )

    model.head.classification_head = (
        RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=3
        )
    )

    return model

# =====================================================
# 测试集
# =====================================================

def get_test_images():

    img_dir = TEST_DIR / "images"

    exts = [".jpg", ".jpeg", ".png", ".bmp"]

    files = []

    for p in img_dir.iterdir():

        if p.suffix.lower() in exts:
            files.append(p)

    files.sort()

    return files


# =====================================================
# torch评测
# =====================================================

def evaluate_torch_detector(
        model,
        weight_path,
        model_name
):

    print(f"\n========== {model_name} ==========")

    model.load_state_dict(
        torch.load(
            weight_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    metric = MeanAveragePrecision()

    times = []

    preds_pc = {
        c: {}
        for c in range(len(CLASS_NAMES))
    }

    gts_pc = {
        c: {}
        for c in range(len(CLASS_NAMES))
    }

    with torch.no_grad():

        for img_path in get_test_images():

            name = img_path.stem

            img = Image.open(img_path).convert("RGB")

            w, h = img.size

            x = TF.to_tensor(img).to(DEVICE)

            t0 = time.time()

            out = model([x])[0]

            times.append(
                time.time() - t0
            )

            keep = out["scores"] >= SCORE_THRESH

            pred_boxes = out["boxes"][keep].cpu()

            pred_scores = out["scores"][keep].cpu()

            # Torch检测器:
            # 1=steel
            # 2=pipe
            # 转YOLO格式
            pred_labels = (
                out["labels"][keep].cpu() - 1
            )

            pred = {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            }

            label_path = (
                TEST_DIR /
                "labels" /
                f"{img_path.stem}.txt"
            )

            gt_boxes, gt_labels = load_yolo_label(
                label_path,
                w,
                h
            )

            target = {
                "boxes": torch.tensor(
                    gt_boxes,
                    dtype=torch.float32
                ),
                "labels": torch.tensor(
                    gt_labels,
                    dtype=torch.int64
                )
            }

            metric.update(
                [pred],
                [target]
            )

            # ------------------
            # P R AP统计
            # ------------------

            for c in range(len(CLASS_NAMES)):
                preds_pc[c][name] = []
                gts_pc[c][name] = []

            for box, cls in zip(
                    gt_boxes,
                    gt_labels
            ):

                gts_pc[int(cls)][name].append(
                    (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3])
                    )
                )

            for box, score, cls in zip(
                    pred_boxes.numpy(),
                    pred_scores.numpy(),
                    pred_labels.numpy()
            ):

                cls = int(cls)

                if cls < 0 or cls >= len(CLASS_NAMES):
                    continue

                preds_pc[cls][name].append(
                    (
                        float(score),
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3])
                    )
                )

    result = metric.compute()

    precs = []
    recs = []

    for c in range(len(CLASS_NAMES)):

        ap, prec, rec, TP, FP, n_gt = compute_ap(
            preds_pc[c],
            gts_pc[c],
            iou_thr=0.5
        )

        precs.append(prec)
        recs.append(rec)
        
    ap_curve = []

    for iou_thr in IOU_LIST:

        aps = []

        for c in range(len(CLASS_NAMES)):

            ap, _, _, _, _, _ = compute_ap(
                preds_pc[c],
                gts_pc[c],
                iou_thr=iou_thr
            )

            aps.append(ap)

        ap_curve.append(
            float(np.mean(aps))
        )
        
    P = float(np.mean(precs))
    R = float(np.mean(recs))

    fps = 1.0 / np.mean(times)

    return {
        "Model": model_name,
        "P": P,
        "R": R,
        "mAP50": float(result["map_50"]),
        "mAP50_95": float(result["map"]),
        "FPS": fps,
        "curve": ap_curve
    }
# =====================================================
# YOLO/RTDETR评测
# =====================================================

def evaluate_ultralytics(weight_path, model_name):

    print(f"\n========== {model_name} ==========")

    if "rtdetr" in model_name.lower():
        model = RTDETR(str(weight_path))
    else:
        model = YOLO(str(weight_path))
    preds_pc = {
        c: {}
        for c in range(len(CLASS_NAMES))
    }

    gts_pc = {
        c: {}
        for c in range(len(CLASS_NAMES))
    }
    metrics = model.val(
        data=str(ROOT / "dataset_all" / "data.yaml"),
        split="test",
        device=DEVICE,
        verbose=False
    )

    times = []

    for img_path in get_test_images():

        name = img_path.stem

        img = Image.open(img_path).convert("RGB")

        w, h = img.size

        t0 = time.time()

        result = model.predict(
            source=str(img_path),
            conf=SCORE_THRESH,
            device=DEVICE,
            verbose=False
        )[0]

        times.append(
            time.time() - t0
        )

        boxes = result.boxes.xyxy.cpu().numpy()

        scores = result.boxes.conf.cpu().numpy()

        labels = result.boxes.cls.cpu().numpy().astype(int)

        for c in range(len(CLASS_NAMES)):
            preds_pc[c][name] = []
            gts_pc[c][name] = []

        label_path = (
            TEST_DIR /
            "labels" /
            f"{img_path.stem}.txt"
        )

        gt_boxes, gt_labels = load_yolo_label(
            label_path,
            w,
            h
        )

        # GT
        for box, cls in zip(
            gt_boxes,
            gt_labels
        ):

            gts_pc[int(cls)][name].append(
                (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3])
                )
            )

        # Pred
        for box, score, cls in zip(
            boxes,
            scores,
            labels
        ):

            preds_pc[int(cls)][name].append(
                (
                    float(score),
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3])
                )
            )

    fps = 1.0 / np.mean(times)
    ap_curve = []

    for iou_thr in IOU_LIST:

        aps = []

        for c in range(len(CLASS_NAMES)):

            ap, _, _, _, _, _ = compute_ap(
                preds_pc[c],
                gts_pc[c],
                iou_thr=iou_thr
            )

            aps.append(ap)

        ap_curve.append(
            float(np.mean(aps))
        )
    return {
        "Model": model_name,
        "P": float(metrics.box.mp),
        "R": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "FPS": fps,
        "curve": ap_curve
    }
# =====================================================
# 画图
# =====================================================

def plot_ap_curves(results):

    # =========================
    # 🔥 全局字体放大设置
    # =========================
    plt.rcParams.update({
        "font.size": 14,          # 基础字体
        "axes.titlesize": 18,     # 标题
        "axes.labelsize": 16,     # x/y轴标签
        "xtick.labelsize": 14,    # x刻度
        "ytick.labelsize": 14,    # y刻度
        "legend.fontsize": 14     # 图例
    })

    plt.figure(figsize=(8, 5))

    markers = {
        "YOLOv8": "o",
        "RT-DETR": "s",
        "Faster R-CNN": "^",
        "RetinaNet": "d"
    }

    for r in results:
        if "curve" not in r:
            continue

        plt.plot(
            IOU_LIST,
            r["curve"],
            marker=markers.get(r["Model"], "o"),
            linewidth=2,
            label = {
                "yolo": "YOLOv8",
                "rtdetr": "RT-DETR",
                "faster_rcnn": "Faster R-CNN",
                "retinanet": "RetinaNet"
            }.get(r["Model"], r["Model"])
        )

    plt.xlabel("IoU Threshold")
    plt.ylabel("Average Precision")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("ap_iou_curve.png", dpi=300)
    plt.show()
# =====================================================
# 主函数
# =====================================================

def main():

    results = []

    results.append(
        evaluate_torch_detector(
            build_frcnn(),
            FRCNN_WEIGHT,
            "Faster R-CNN"
        )
    )

    results.append(
        evaluate_torch_detector(
            build_retinanet(),
            RETINANET_WEIGHT,
            "RetinaNet"
        )
    )
    results.append(
        evaluate_ultralytics(
            YOLO_WEIGHT,
            "YOLOv8"
        )
    )

    results.append(
        evaluate_ultralytics(
            RTDETR_WEIGHT,
            "RT-DETR"
        )
    )
    

    print("\n==============================")
    print("Benchmark Results")
    print("==============================")

    for r in results:

        print(
            f"{r['Model']:12s} "
            f"P={r['P']:.4f} "
            f"R={r['R']:.4f} "
            f"mAP50={r['mAP50']:.4f} "
            f"mAP50-95={r['mAP50_95']:.4f} "
            f"FPS={r['FPS']:.2f}"
        )
        
    save_results = []

    for r in results:

        rr = r.copy()

        rr.pop("curve")

        save_results.append(rr)
    with open(
        "benchmark_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(save_results[0].keys())
        )

        writer.writeheader()

        for r in save_results:
            writer.writerow(r)

    print("\n结果已保存:")
    print("benchmark_results.csv")
    plot_ap_curves(results)

    print(
        "AP曲线已保存: ap_iou_curve.png"
    )


if __name__ == "__main__":
    main()