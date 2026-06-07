# P-GPR: A Ground Penetrating Radar Dataset for Pipeline Detection

P-GPR is a real-world Ground Penetrating Radar (GPR) dataset for **pipeline
detection** in indoor construction scenarios. It provides B-scan images with
manually annotated bounding boxes for pipeline hyperbolic reflections and for
reinforcement (rebar) clutter, together with relatively accurate ground-truth
pipeline locations obtained during the construction period.

The dataset accompanies the CIKM '26 resource paper *"P-GPR: a Ground
Penetrating Radar Dataset for Pipeline Detection"*. If you use it, please cite
the paper (see [`citation.bib`](citation.bib)).

---

## 1. Motivation

Open-source GPR datasets for pipeline detection are scarce, small, and usually
lack reliable ground truth — forcing researchers to rely on synthetic data or
private datasets, which hurts reproducibility and method evaluation. P-GPR
addresses this with a two-stage collection protocol:

1. **Ground-truth stage** — during pipeline installation, while the pipes were
   still exposed, their true embedded locations were recorded with reference to
   the construction drawings.
2. **GPR acquisition stage** — after wall leveling and tiling were completed
   (pipes now unseen), B-scan data were collected over the same regions.

Because the pipe locations were fixed *before* the radar scan, the bounding-box
annotations are anchored to verified ground truth rather than to subjective
interpretation of the echoes alone.

---

## 2. Acquisition Setup

| Property | Value |
|---|---|
| Site | Indoor renovation construction site, Beijing, China |
| GPR devices | 2 devices (to capture cross-device feature variation) |
| Central frequency | 1600 MHz |
| Time window | 8 ns |
| Pipeline types | PPR water pipes, PVC electrical conduits |
| Wall types | Reinforced-concrete walls, lightweight brick walls |
| Burial configurations | Single-pipe and multiple-pipe arrangements |

For most pipeline-embedded regions, 2–3 survey lines were acquired at different
heights with both devices. Several pipe-free survey lines over reinforced
concrete were also collected as reference samples, since rebar echoes closely
resemble pipeline echoes.

---

## 3. Dataset Contents

### 3.1 Format

Images are 8-bit grayscale **B-scan** PNGs. Annotations are in **YOLO** format
(one `.txt` per image; each line `class cx cy w h`, normalized to `[0, 1]`),
created with [LabelImg](https://github.com/HumanSignal/labelImg). The vertex and
both endpoints of each hyperbolic echo are enclosed as tightly as possible.

The image **height is fixed at 256 px**, while the **width varies (95–3081 px)**:
the horizontal axis is spatial and the aspect ratio is kept proportional to the
survey-line length (a 2 m line yields twice the width of a 1 m line). Users may
crop images to a fixed horizontal length for augmentation.

### 3.2 Classes

| ID | Name (`data.yaml`) | Meaning |
|---|---|---|
| 0 | `steel` | Reinforcement / rebar reflection (clutter) |
| 1 | `pipe`  | Pipeline (PPR water pipe / PVC conduit) reflection |

### 3.3 Statistics

| Split | Images | Ratio |
|---|---|---|
| train | 191 | ~70% |
| val   | 56  | ~20% |
| test  | 30  | ~10% |
| **Total** | **277** | 100% |

Annotated bounding boxes: **1,504** total — **381** `pipe` + **1,123** `steel`.

### 3.4 Directory Layout

```
dataset_all/
├── data.yaml              # Ultralytics-style config (paths, nc=2, names)
├── README.md
├── LICENSE
├── citation.bib
├── benchmark_results.xlsx # baseline results (this release)
├── train/
│   ├── images/   *.png
│   └── labels/   *.txt
├── val/
│   ├── images/   *.png
│   └── labels/   *.txt
└── test/
    ├── images/   *.png
    └── labels/   *.txt
```

`data.yaml`:

```yaml
path: <dataset root>
train: train/images
val: val/images
test: test/images
nc: 2
names:
  0: steel
  1: pipe
```

> Edit the `path` field in `data.yaml` to your local dataset root before training.

---

## 4. Quick Start

Train a YOLOv8 baseline with [Ultralytics](https://github.com/ultralytics/ultralytics):

```bash
pip install ultralytics
yolo detect train data=dataset_all/data.yaml model=yolov8n.pt imgsz=640 epochs=300 patience=20
yolo detect val   data=dataset_all/data.yaml model=runs/detect/train/weights/best.pt split=test
```

---

## 5. Baseline Results

Four widely used detectors were trained on the combined two-device data using a
unified early-stopping rule (stop after 20 epochs without validation
improvement). Full numbers are in [`benchmark_results.xlsx`](benchmark_results.xlsx);
the per-IoU AP curves are in `ap_iou_curve.png`.

| Model | Precision | Recall | mAP@0.5 | mAP@[0.5:0.95] | FPS |
|---|---|---|---|---|---|
| Faster R-CNN | 0.752 | 0.968 | 0.917 | 0.507 | 18.0 |
| RetinaNet    | 0.745 | 0.947 | 0.889 | 0.499 | 37.2 |
| YOLOv8n      | 0.880 | 0.875 | 0.919 | 0.497 | 41.2 |
| RT-DETR-L    | 0.335 | 0.771 | 0.509 | 0.265 | 31.3 |

**Observations.** Faster R-CNN, RetinaNet and YOLOv8 all reach satisfactory
mAP@0.5, confirming the dataset is suitable for benchmarking pipeline detection
despite the high noise of real GPR data. All methods degrade sharply at higher
IoU thresholds, showing that precise localization of buried pipelines without
domain priors remains an open problem.

---

## 6. Intended Use & Limitations

- **Intended use.** Training and benchmarking object detection methods for GPR
  pipeline localization; studying cross-device generalization; distinguishing
  pipeline echoes from rebar clutter in reinforced-concrete walls.
- **Limitations.** Collected at a single indoor Beijing site with two 1600 MHz
  devices; pipe materials limited to PPR and PVC. Generalization to outdoor
  scenes, other frequencies, or other pipe materials is not guaranteed.

---

## 7. License

Released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
You may share and adapt the data, including commercially, provided you give
appropriate credit and cite the paper. See [`LICENSE`](LICENSE).

---

## 8. Citation

See [`citation.bib`](citation.bib).

```bibtex
@inproceedings{yue2026pgpr,
  title     = {P-GPR: a Ground Penetrating Radar Dataset for Pipeline Detection},
  author    = {Yue, Jiaqin and Qin, Yingrong and Zhu, Menghao and Ye, Shengbo and Wang, Yue},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  address   = {Rome, Italy},
  publisher = {ACM}
}
```

---

## 9. Acknowledgments

This work is supported by the National Key Research and Development Program of
China under Grant No. 2022YFF0606900.
