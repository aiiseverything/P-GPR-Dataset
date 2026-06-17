# -*- coding: utf-8 -*-
"""
Baseline 2: Faster R-CNN on YOLO-format GPR B-scan dataset.
- 数据: dataset/{train,val}/{images,labels}  (YOLO 标注, 2 类: 0=steel, 1=pipe)
- 训练 + 评估一体；评估同时报告 bbox mAP@0.5 与顶点定位 (P/R/MAE_x/MAE_y)
用法：
  python baseline_faster_rcnn.py train --data dataset --epochs 20 --out runs/frcnn
  python baseline_faster_rcnn.py eval  --data dataset --weights runs/frcnn/best.pth
"""
import os, csv, math, argparse, json, time
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from _baseline_utils import (list_images, load_yolo_label, yolo_label_path_for_image,
                             compute_ap, vertex_eval, depth_per_pixel)

CLASS_NAMES = ['steel', 'pipe']  # YOLO id 0,1 -> model labels 1,2


class GPRYoloDataset(data.Dataset):
    def __init__(self, root):
        self.img_dir = os.path.join(root, 'images')
        self.paths = list_images(self.img_dir)
        # 过滤掉无标注或空标注
        self.items = []
        for p in self.paths:
            with Image.open(p) as im:
                W, H = im.size
            labs = load_yolo_label(yolo_label_path_for_image(p), W, H)
            labs = [l for l in labs if l[3] > l[1] and l[4] > l[2]]
            if labs:
                self.items.append((p, W, H, labs))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, W, H, labs = self.items[idx]
        img = Image.open(path).convert('RGB')
        boxes = torch.tensor([[l[1], l[2], l[3], l[4]] for l in labs], dtype=torch.float32)
        labels = torch.tensor([l[0] + 1 for l in labs], dtype=torch.int64)  # +1: bg=0
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
        return TF.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes=3):  # bg + steel + pipe
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model
def evaluate_map(model, dataset, device, score_thresh=0.5):
    """
    Validation mAP@0.5
    """
    model.eval()

    preds_pc = {c: {} for c in range(len(CLASS_NAMES))}
    gts_pc = {c: {} for c in range(len(CLASS_NAMES))}

    with torch.no_grad():

        for idx in range(len(dataset)):

            img, target = dataset[idx]
            name = str(idx)

            out = model([img.to(device)])[0]

            scores = out['scores'].cpu().numpy()
            boxes = out['boxes'].cpu().numpy()
            labels = out['labels'].cpu().numpy()

            keep = scores >= score_thresh

            for c in range(len(CLASS_NAMES)):
                preds_pc[c][name] = []
                gts_pc[c][name] = []

            gt_boxes = target['boxes'].numpy()
            gt_labels = target['labels'].numpy()

            for b, l in zip(gt_boxes, gt_labels):
                gts_pc[int(l) - 1][name].append(
                    (float(b[0]), float(b[1]),
                     float(b[2]), float(b[3]))
                )

            for s, b, l in zip(
                scores[keep],
                boxes[keep],
                labels[keep]
            ):
                preds_pc[int(l) - 1][name].append(
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

    return float(np.mean(aps))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}  amp={args.amp}  num_workers={args.num_workers}')
    train_ds = GPRYoloDataset(os.path.join(args.data, 'train'))
    val_ds = GPRYoloDataset(os.path.join(args.data, 'val'))
    print(f'[data] train={len(train_ds)} val={len(val_ds)}')
    loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=args.num_workers,
                             pin_memory=(device.type == 'cuda'))
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=max(args.epochs // 3, 1), gamma=0.5)
    scaler = None
    if args.amp and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    os.makedirs(args.out, exist_ok=True)

    best_map = 0.0
    early_stop_counter = 0
    min_delta = 1e-4
    log = []
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0; t0 = time.time()
        for imgs, targets in loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            optim.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = model(imgs, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:

                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())

                loss.backward()
                optim.step()
            ep_loss += loss.item()
        sched.step()
        ep_loss /= max(len(loader), 1)

        val_map = evaluate_map(
            model,
            val_ds,
            device,
            score_thresh=0.5
        )

        dt = time.time() - t0

        print(
            f'Epoch {ep+1}/{args.epochs} '
            f'loss={ep_loss:.4f} '
            f'val_mAP={val_map:.4f} '
            f'({dt:.1f}s)',
            flush=True
        )
        log.append({
            'epoch': ep + 1,
            'loss': float(ep_loss),
            'val_map': float(val_map),
            'sec': float(dt)
        })
        if val_map > best_map + min_delta:

            best_map = val_map
            early_stop_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(args.out, 'best.pth')
            )

            print(
                f'Best model updated '
                f'(mAP={best_map:.4f})'
            )

        else:

            early_stop_counter += 1

            print(
                f'No improvement '
                f'({early_stop_counter}/{args.patience})'
            )

            if early_stop_counter >= args.patience:

                print(
                    f'\nEarly stopping triggered! '
                    f'Best mAP={best_map:.4f}'
                )

                break
    torch.save(
    model.state_dict(),
    os.path.join(args.out, 'last.pth')
)
    with open(os.path.join(args.out, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print('saved ->', args.out)


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    val_dir = os.path.join(args.data, 'val', 'images')
    paths = list_images(val_dir)

    # per-class containers
    preds_pc = {c: {} for c in range(len(CLASS_NAMES))}
    gts_pc = {c: {} for c in range(len(CLASS_NAMES))}
    # vertex localization (pipe class only, since pipes are typical hyperbola targets)
    vtx_TP = vtx_FP = vtx_FN = 0
    err_x = []; err_y = []
    per_image_rows = []

    with torch.no_grad():
        for p in paths:
            name = os.path.basename(p)
            img = Image.open(p).convert('RGB')
            W, H = img.size
            dpp = depth_per_pixel(H, args.time_window, args.epsilon_r)

            gts = load_yolo_label(yolo_label_path_for_image(p), W, H)
            for c in range(len(CLASS_NAMES)):
                gts_pc[c][name] = [(g[1], g[2], g[3], g[4]) for g in gts if g[0] == c]

            x = TF.to_tensor(img).to(device)
            out = model([x])[0]
            scores = out['scores'].cpu().numpy()
            boxes = out['boxes'].cpu().numpy()
            labels = out['labels'].cpu().numpy()  # 1=steel, 2=pipe
            keep = scores >= args.score_thresh

            for c in range(len(CLASS_NAMES)):
                preds_pc[c][name] = []
            for s, b, l in zip(scores[keep], boxes[keep], labels[keep]):
                c = int(l) - 1
                if 0 <= c < len(CLASS_NAMES):
                    preds_pc[c][name].append((float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3])))

            # vertex localization: use ALL classes combined (任何管线/钢筋目标的 bbox 顶部中点)
            preds_m = []
            for s, b, l in zip(scores[keep], boxes[keep], labels[keep]):
                ax = (b[0] + b[2]) / 2.0
                ay = b[1]
                preds_m.append((ax * args.trace_spacing, ay * dpp))
            gts_m = []
            for g in gts:
                ax = (g[1] + g[3]) / 2.0
                ay = g[2]
                gts_m.append((ax * args.trace_spacing, ay * dpp))
            TP, FP, FN, pairs = vertex_eval(preds_m, gts_m, args.match_thresh)
            vtx_TP += TP; vtx_FP += FP; vtx_FN += FN
            err_x += [abs(p_[0] - p_[2]) for p_ in pairs]
            err_y += [abs(p_[1] - p_[3]) for p_ in pairs]
            per_image_rows.append((name, len(preds_m), len(gts_m), TP, FP, FN))

    print('\n=== Faster R-CNN: bbox mAP@0.5 ===')
    aps = []
    for c, cname in enumerate(CLASS_NAMES):
        ap, prec, rec, TP, FP, n_gt = compute_ap(preds_pc[c], gts_pc[c], iou_thr=0.5)
        aps.append(ap)
        print(f'  [{cname}] AP={ap:.3f}  P={prec:.3f}  R={rec:.3f}  TP={TP} FP={FP} GT={n_gt}')
    print(f'  mAP@0.5 = {np.mean(aps):.3f}')

    print('\n=== Faster R-CNN: 顶点定位 ===')
    R = vtx_TP / max(vtx_TP + vtx_FN, 1)
    P = vtx_TP / max(vtx_TP + vtx_FP, 1)
    mx = float(np.mean(err_x)) if err_x else float('nan')
    my = float(np.mean(err_y)) if err_y else float('nan')
    print(f'  TP={vtx_TP} FP={vtx_FP} FN={vtx_FN}  P={P:.3f} R={R:.3f}  MAE_x={mx:.3f} m  MAE_y={my:.3f} m')

    with open(args.out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'n_pred', 'n_gt', 'TP', 'FP', 'FN'])
        w.writerows(per_image_rows)
    print('saved per-image ->', args.out_csv)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', default='dataset')
    t.add_argument('--epochs', type=int, default=20)
    t.add_argument('--batch_size', type=int, default=2)
    t.add_argument('--lr', type=float, default=5e-3)
    t.add_argument('--out', default='runs/frcnn')
    t.add_argument('--num_workers', type=int, default=4)
    t.add_argument('--amp', action='store_true', help='Enable CUDA mixed precision')
    t.add_argument( '--patience', type=int,default=10, help='Early stopping patience')
    e = sub.add_parser('eval')
    e.add_argument('--data', default='dataset')
    e.add_argument('--weights', required=True)
    e.add_argument('--trace_spacing', type=float, default=0.002076)
    e.add_argument('--time_window', type=float, default=8e-9)
    e.add_argument('--epsilon_r', type=float, default=4.0)
    e.add_argument('--match_thresh', type=float, default=0.10)
    e.add_argument('--score_thresh', type=float, default=0.5)
    e.add_argument('--out_csv', default='baseline_frcnn_results.csv')
    args = ap.parse_args()
    (train if args.cmd == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
