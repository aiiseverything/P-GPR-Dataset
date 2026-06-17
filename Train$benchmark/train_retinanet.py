# -*- coding: utf-8 -*-

import os
import json
import time
import argparse

import torch
import torch.utils.data as data

from PIL import Image

import torchvision.transforms.functional as TF

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from _baseline_utils import (
    list_images,
    load_yolo_label,
    yolo_label_path_for_image,
    compute_ap
)

CLASS_NAMES = ["steel", "pipe"]


# =====================================================
# Dataset
# =====================================================

class GPRYoloDataset(data.Dataset):

    def __init__(self, root):

        self.img_dir = os.path.join(root, "images")

        self.paths = list_images(self.img_dir)

        self.items = []

        for p in self.paths:

            with Image.open(p) as im:
                W, H = im.size

            labs = load_yolo_label(
                yolo_label_path_for_image(p),
                W,
                H
            )

            labs = [
                l for l in labs
                if l[3] > l[1] and l[4] > l[2]
            ]

            if labs:
                self.items.append(
                    (p, W, H, labs)
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        path, W, H, labs = self.items[idx]

        img = Image.open(path).convert("RGB")

        boxes = torch.tensor(
            [[l[1], l[2], l[3], l[4]] for l in labs],
            dtype=torch.float32
        )

        labels = torch.tensor(
            [l[0] + 1 for l in labs],
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return TF.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


# =====================================================
# Model
# =====================================================

def build_model():

    model = retinanet_resnet50_fpn(
        weights="DEFAULT"
    )

    num_classes = 3

    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    return model


# =====================================================
# mAP50
# =====================================================

def evaluate_map(
    model,
    dataset,
    device,
    score_thresh=0.5
):

    model.eval()

    preds_pc = {
        c: {}
        for c in range(len(CLASS_NAMES))
    }

    gts_pc = {
        c: {}
        for c in range(len(CLASS_NAMES))
    }

    with torch.no_grad():

        for idx in range(len(dataset)):

            img, target = dataset[idx]

            name = str(idx)

            out = model(
                [img.to(device)]
            )[0]

            scores = out["scores"].cpu().numpy()
            boxes = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()

            keep = scores >= score_thresh

            for c in range(len(CLASS_NAMES)):
                preds_pc[c][name] = []
                gts_pc[c][name] = []

            gt_boxes = target["boxes"].numpy()
            gt_labels = target["labels"].numpy()

            for b, l in zip(
                gt_boxes,
                gt_labels
            ):

                gts_pc[int(l)-1][name].append(
                    (
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                        float(b[3])
                    )
                )

            for s, b, l in zip(
                scores[keep],
                boxes[keep],
                labels[keep]
            ):

                preds_pc[int(l)-1][name].append(
                    (
                        float(s),
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                        float(b[3])
                    )
                )

    aps = []

    for c in range(len(CLASS_NAMES)):

        ap, _, _, _, _, _ = compute_ap(
            preds_pc[c],
            gts_pc[c],
            iou_thr=0.5
        )

        aps.append(ap)

    model.train()

    return float(sum(aps) / len(aps))


# =====================================================
# Train
# =====================================================

def train(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_ds = GPRYoloDataset(
        os.path.join(args.data, "train")
    )

    val_ds = GPRYoloDataset(
        os.path.join(args.data, "val")
    )

    print(
        f"train={len(train_ds)} "
        f"val={len(val_ds)}"
    )

    loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    model = build_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(args.epochs // 3, 1),
        gamma=0.5
    )

    os.makedirs(args.out, exist_ok=True)

    best_map = 0.0
    patience_counter = 0

    log = []

    for epoch in range(args.epochs):

        model.train()

        epoch_loss = 0

        t0 = time.time()

        for imgs, targets in loader:

            imgs = [
                x.to(device)
                for x in imgs
            ]

            targets = [
                {
                    k: v.to(device)
                    for k, v in t.items()
                }
                for t in targets
            ]

            loss_dict = model(
                imgs,
                targets
            )

            loss = sum(
                loss_dict.values()
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        epoch_loss /= len(loader)

        val_map = evaluate_map(
            model,
            val_ds,
            device
        )

        dt = time.time() - t0

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"loss={epoch_loss:.4f} "
            f"val_mAP={val_map:.4f} "
            f"time={dt:.1f}s"
        )

        log.append({
            "epoch": epoch+1,
            "loss": epoch_loss,
            "val_map": val_map
        })

        if val_map > best_map:

            best_map = val_map

            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(
                    args.out,
                    "best.pth"
                )
            )

            print(
                f"Best updated "
                f"{best_map:.4f}"
            )

        else:

            patience_counter += 1

        if patience_counter >= args.patience:

            print(
                f"Early stopping "
                f"at epoch {epoch+1}"
            )

            break

    torch.save(
        model.state_dict(),
        os.path.join(
            args.out,
            "last.pth"
        )
    )

    with open(
        os.path.join(
            args.out,
            "train_log.json"
        ),
        "w"
    ) as f:

        json.dump(
            log,
            f,
            indent=2
        )


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="dataset"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--out",
        default="runs/retinanet"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20
    )

    args = parser.parse_args()

    train(args)