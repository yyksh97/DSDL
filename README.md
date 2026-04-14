# DSDL: Distance-guided, Signed, and Densified Learning for Tiny Object Detection

Official implementation of **"Tiny Object Detection Using Distance-guided, Signed, and Densified Learning (DSDL) for Construction Site Safety Monitoring"** — *Automation in Construction* (2026). [[paper]](https://www.sciencedirect.com/science/article/pii/S0926580526001706)

DSDL is a **model-agnostic enhancement** for one-stage object detectors that addresses the *Minnow Net Problem*: tiny objects escaping detection due to coarse anchor intervals, positive-only distribution bins, and low binning resolution. DSDL improves tiny object detection **without modifying the underlying model architecture** and is compatible with any detector employing Task Alignment Learning (TAL) and Distribution Focal Loss (DFL).

Built on top of [YOLOv9](https://github.com/WongKinYiu/yolov9).

---

## Contents

- [Method Overview](#method-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Binning configurations](#binning-configurations)
- [Training](#training)
- [Validation](#validation)
- [Inference / Detection](#inference--detection)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Citation](#citation)
- [한국어 설명](#한국어-설명)

---

## Method Overview

DSDL consists of three components that together resolve the three "nets" of the Minnow Net Problem:

| Component | Problem Addressed | How |
|---|---|---|
| **D-TAL** *(Distance-guided TAL)* | **Spatial net** — GT boxes smaller than anchor strides get zero positive anchors under vanilla TAL. | For each GT, if fewer than τ anchors fall inside, supplement with the closest IoU>0 anchors. Controlled by `--dtal` and the `YOLOM` env var (τ). |
| **S-DFL** *(Signed DFL)* | **Range net** — Vanilla DFL bins are non-negative, so boundaries beyond the GT cannot be represented. | Extend the DFL bin list to negative values. Configured via `--reg_list`. |
| **D-DFL** *(Densified DFL)* | **Quantization net** — Uniform integer bins collapse to bins 0–1 for tiny objects, losing the probabilistic nature of DFL. | Use finer-grained non-uniform bins near zero. Configured via `--reg_list`. |

Because S-DFL and D-DFL are both expressed through the bin list, you compose any combination by setting `--reg_list` accordingly (see [Binning configurations](#binning-configurations)).

---

## Installation

```bash
git clone https://github.com/yyksh97/DSDL.git
cd DSDL
pip install -r requirements.txt
```

Python ≥ 3.8, PyTorch ≥ 1.7. GPU with CUDA strongly recommended for training.

---

## Pretrained Weights

DSDL fine-tunes on top of standard YOLOv9 / GELAN backbones. Download the base weights from the official [YOLOv9 Releases](https://github.com/WongKinYiu/yolov9/releases):

| Model | Weights |
|---|---|
| YOLOv9-T | [yolov9-t-converted.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt) |
| YOLOv9-S | [yolov9-s.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt) |
| YOLOv9-M | [yolov9-m.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt) |
| YOLOv9-C | [yolov9-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) |
| YOLOv9-E | [yolov9-e.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt) |
| GELAN-S | [gelan-s.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-s.pt) |
| GELAN-M | [gelan-m.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt) |
| GELAN-C | [gelan-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt) |
| GELAN-E | [gelan-e.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt) |
| **YOLOv9-C-P2** (ours, MS COCO pretrained) | [yolov9-c-p2.pt](./yolov9-c-p2.pt) |

---

## Quick Start

Usage is fully compatible with the original YOLOv9 — to enable DSDL, simply add `--dtal` and a `--reg_list` configuration as shown below.

Train YOLOv9-C with the full DSDL (D-TAL + S-DFL + D-DFL) on a custom dataset at 640px:

```bash
python train_dual.py \
    --cfg      models/detect/yolov9-c.yaml \
    --data     path/to/your_dataset.yaml \
    --hyp      data/hyps/hyp.scratch-high.yaml \
    --weights  yolov9-c-converted.pt \
    --imgsz    640 \
    --batch    16 \
    --epochs   50 \
    --device   0 \
    --dtal \
    --reg_list -2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

---

## Binning configurations

DSDL controls DFL quantization entirely via `--reg_list` (the list of bin center values). The paper uses the following configurations:

| Setting | `--reg_list` | # Bins |
|---|---|---|
| **Standard DFL** (baseline) | `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 17 |
| **S-DFL** | `-2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 19 |
| **D-DFL** | `0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 21 |
| **S-DFL + D-DFL** | `-2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 27 |

Add `--dtal` to activate **D-TAL** on top of any of the above.

---

## Training

`train_dual.py` supports any model config whose detection head is one of **`Detect`, `DDetect`, `DualDetect`, `DualDDetect`, `TripleDetect`, `TripleDDetect`**. The DSDL flags transfer automatically regardless of the chosen YAML.

### Common recipe (from the paper, Section 4.4)

```bash
python train_dual.py \
    --cfg      models/detect/yolov9-c.yaml \
    --data     data/visdrone.yaml \
    --hyp      data/hyps/hyp.scratch-high.yaml \
    --weights  yolov9-c-converted.pt \
    --imgsz    640 \
    --batch    4 \
    --epochs   50 \
    --close-mosaic 15 \
    --device   0 \
    --dtal \
    --reg_list -2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

### YOLOv9c-P2 (for ultra-tiny objects)

The paper also evaluates a YOLOv9c variant with an added P2 feature layer for higher spatial resolution. The config ships at [models/detect/yolov9-c-p2.yaml](models/detect/yolov9-c-p2.yaml) and is trained in the same way.

### DSDL CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--dtal` | off | Enable D-TAL (distance-supplemented anchor assignment). |
| `--reg_list` | `0 1 … 15` | DFL bin centers. Set negatives for S-DFL, fractional for D-DFL, or both. |

### τ hyperparameter

The TAL target candidate count τ (Algorithm 1 in the paper) is configured via the `YOLOM` environment variable, inherited from the YOLOv9 baseline:

```bash
YOLOM=13 python train_dual.py --dtal --reg_list ...   # paper default τ=13
YOLOM=4  python train_dual.py --dtal --reg_list ...   # ablation τ=4
YOLOM=20 python train_dual.py --dtal --reg_list ...   # ablation τ=20
```

### Multi-GPU training

```bash
python -m torch.distributed.run --nproc_per_node 2 --master_port 9527 \
    train_dual.py \
    --cfg models/detect/yolov9-c.yaml \
    --data data/your.yaml \
    --device 0,1 --sync-bn \
    --dtal --reg_list ...
```

---

## Validation

Validate a trained checkpoint:

```bash
python val_dual.py \
    --data    data/your_dataset.yaml \
    --weights runs/train/<your_run>/weights/best.pt \
    --imgsz   640 \
    --batch   1 \
    --conf    0.001 \
    --iou     0.7 \
    --device  0
```

The validator prints mAP@50, mAP@50:95, and per-class metrics. No DSDL-specific flags are required at validation time — bin information is baked into the checkpoint.

---

## Inference / Detection

Run on images or videos:

```bash
python detect_dual.py \
    --source  path/to/images_or_video \
    --weights runs/train/<your_run>/weights/best.pt \
    --imgsz   640 \
    --conf    0.25 \
    --iou     0.45 \
    --device  0
```

Useful options:
- `--save-txt` — save YOLO-format labels alongside detections
- `--save-conf` — include confidence in saved labels
- `--view-img` — preview live (desktop)

---

## Reproducing Paper Results

The paper reports ablations on the **YKH** (construction-site PPE) and **VisDrone** datasets. To reproduce an ablation row:

1. Pretrain YOLOv9-c (or YOLOv9-c-P2) on MS COCO 2017 with standard YOLOv9 settings (`--epochs 500 --close-mosaic 15 --batch-size 36`).
2. Fine-tune 50 epochs with the target binning config and τ:

```bash
# Full DSDL on VisDrone @ 640px, τ=13
YOLOM=13 python train_dual.py \
    --cfg     models/detect/yolov9-c.yaml \
    --data    data/visdrone.yaml \
    --weights runs/pretrain/yolov9-c/weights/last.pt \
    --imgsz 640 --batch 4 --epochs 50 --close-mosaic 15 \
    --dtal \
    --reg_list -2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

Swap `--reg_list` and `--dtal` per the [binning table](#binning-configurations) to reproduce specific rows.

> **Note on pretrained weights.** When `--reg_list` changes the number of bins, detection-head channels (`4 * reg_max`) change too. The backbone transfers cleanly from vanilla YOLOv9 checkpoints; only the final detection-head layer's output nodes change and are re-initialized. This is expected behavior per the paper (Section 3). The affected parameters are an extremely small fraction of the total model, so the impact on training and inference is negligible.

---

## Citation

If you use DSDL in your research, please cite:

```bibtex
@article{KIM2026106929,
title = {Tiny object detection using Distance-guided, Signed, and Densified Learning (DSDL) for construction site safety monitoring},
journal = {Automation in Construction},
volume = {187},
pages = {106929},
year = {2026},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2026.106929},
url = {https://www.sciencedirect.com/science/article/pii/S0926580526001706},
author = {Seokhwan Kim and Taegeon Kim and Kichang Choi and Siheon Joo and Hongjo Kim},
keywords = {Tiny object detection, Model-agnostic enhancement, Distance-guided Task Alignment Learning, Signed Distribution Focal Loss, Densified Distribution Focal Loss},
abstract = {Ensuring construction safety requires accurate detection of personal protective equipment (PPE) to prevent major accidents. However, PPE items such as hooks and straps are typically extremely small (fewer than 162 pixels2), making them difficult to detect with conventional computer vision models. This paper identifies fundamental limitations in object detection, termed the Minnow Net Problem, in which tiny objects escape detection due to coarse anchor intervals, positive-only distribution bins, and low binning resolution. To address these challenges, this paper introduces Distance-guided Task Alignment Learning (D-TAL), Signed Distribution Focal Loss (S-DFL), and Densified Distribution Focal Loss (D-DFL) Learning (DSDL), a set of techniques that enhance tiny object detection without requiring modifications to model architectures. Experimental results demonstrate substantial improvements, achieving up to a 48.6 percentage-point gain in tiny object detection while preserving inference speed. DSDL functions as a model-agnostic enhancement applicable to modern one-stage object detection models, and the source code is publicly accessible.}
}
```

---

## Contact

For questions or issues regarding this implementation or the paper, please contact:

- Seokhwan Kim — [yyksh2019@yonsei.ac.kr](mailto:yyksh2019@yonsei.ac.kr)
- Hongjo Kim (corresponding author) — [hongjo@yonsei.ac.kr](mailto:hongjo@yonsei.ac.kr)

---

## Acknowledgements

DSDL is implemented on top of [YOLOv9 by Wang, Yeh, and Liao (2024)](https://github.com/WongKinYiu/yolov9). The D-TAL design extends Task Alignment Learning from [TOOD (Feng et al., 2021)](https://arxiv.org/abs/2108.07755), and the DFL reformulation builds on [Generalized Focal Loss (Li et al., 2020)](https://arxiv.org/abs/2006.04388).

---

# 한국어 설명

논문 **"Tiny Object Detection Using Distance-guided, Signed, and Densified Learning (DSDL) for Construction Site Safety Monitoring"** (*Automation in Construction*, 2026)의 공식 구현 레포입니다. [[논문 링크]](https://www.sciencedirect.com/science/article/pii/S0926580526001706)

DSDL은 **모델 구조를 바꾸지 않고** 기존 1-stage 객체 탐지기의 **tiny object 검출 성능**을 끌어올리는 **model-agnostic** 기법입니다. TAL(Task Alignment Learning)과 DFL(Distribution Focal Loss)을 사용하는 모든 검출기에 바로 적용하여 쓸 수 있습니다.

이 코드는 [YOLOv9](https://github.com/WongKinYiu/yolov9)을 베이스로 구현하였습니다. 

---

## 방법 개요

DSDL은 *Minnow Net Problem*(극소 객체가 성긴 그물을 빠져나가듯 검출되지 못하는 현상)을 세 가지 관점에서 해결합니다:

| 컴포넌트 | 해결하는 문제 | 구현 방식 |
|---|---|---|
| **D-TAL** | **공간 그물** — GT 박스가 앵커 간격보다 작으면 TAL 상 positive 앵커가 0이 되는 문제 | GT 당 내부 앵커가 τ보다 적으면, 가장 가까운 IoU>0 앵커로 보충. `--dtal`과 `YOLOM` 환경변수(τ)로 제어. |
| **S-DFL** | **범위 그물** — DFL bin이 양수만이라 GT 밖 경계를 표현 못함 | DFL bin 리스트를 음수까지 확장. `--reg_list`로 설정. |
| **D-DFL** | **양자화 그물** — 균등 정수 bin은 극소 객체에서 0~1 bin에 쏠려 DFL의 확률적 표현력이 사라지는 문제 | 0 근처에 더 촘촘한 비균등 bin을 배치. `--reg_list`로 설정. |

S-DFL과 D-DFL은 모두 `--reg_list` 하나로 표현되므로, 조합에 따라 bin 리스트만 바꾸면 됩니다.

---

## 설치

```bash
git clone https://github.com/yyksh97/DSDL.git
cd DSDL
pip install -r requirements.txt
```

Python ≥ 3.8, PyTorch ≥ 1.7. 학습에는 CUDA GPU 권장.

---

## 사전학습 가중치

DSDL은 표준 YOLOv9 / GELAN 백본 위에서 fine-tuning합니다. 기본 가중치는 공식 [YOLOv9 Releases](https://github.com/WongKinYiu/yolov9/releases)에서 받을 수 있습니다:

| 모델 | 가중치 |
|---|---|
| YOLOv9-T | [yolov9-t-converted.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt) |
| YOLOv9-S | [yolov9-s.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt) |
| YOLOv9-M | [yolov9-m.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt) |
| YOLOv9-C | [yolov9-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) |
| YOLOv9-E | [yolov9-e.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt) |
| GELAN-S | [gelan-s.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-s.pt) |
| GELAN-M | [gelan-m.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt) |
| GELAN-C | [gelan-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt) |
| GELAN-E | [gelan-e.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt) |
| **YOLOv9-C-P2** (본 저장소, MS COCO 사전학습) | [yolov9-c-p2.pt](./yolov9-c-p2.pt) |

---

## 빠른 시작

기본적으로 YOLOv9과 완벽히 동일한 방법으로 사용할 수 있습니다. DSDL을 적용하려면 아래와 같이 --dtal과 --reg_list 설정을 사용하면됩니다.

YOLOv9-C에 전체 DSDL(D-TAL + S-DFL + D-DFL)을 적용해 640px 해상도로 학습:

```bash
python train_dual.py \
    --cfg      models/detect/yolov9-c.yaml \
    --data     path/to/your_dataset.yaml \
    --hyp      data/hyps/hyp.scratch-high.yaml \
    --weights  yolov9-c-converted.pt \
    --imgsz    640 \
    --batch    16 \
    --epochs   50 \
    --device   0 \
    --dtal \
    --reg_list -2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

---

## Binning 설정

DSDL의 DFL 양자화는 전부 `--reg_list`(bin 중심값 리스트)로 제어합니다. 논문에서는 다음과 같은 설정을 사용하였습니다. 

| 세팅 | `--reg_list` | bin 수 |
|---|---|---|
| **Standard DFL** (baseline) | `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 17 |
| **S-DFL** | `-2 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 19 |
| **D-DFL** | `0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 21 |
| **S-DFL + D-DFL** | `-2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 27 |

여기에 `--dtal` 플래그를 더하면 **D-TAL**까지 함께 활성화됩니다.

---

## 학습

`train_dual.py`는 detection head가 **`Detect`, `DDetect`, `DualDetect`, `DualDDetect`, `TripleDetect`, `TripleDDetect`** 중 하나인 모든 YAML을 지원합니다. 어떤 모델 YAML을 쓰든 동일한 DSDL 설정값을 사용하면 됩니다. 

### 논문 재현 레시피 (4.4절)

```bash
python train_dual.py \
    --cfg      models/detect/yolov9-c.yaml \
    --data     data/visdrone.yaml \
    --hyp      data/hyps/hyp.scratch-high.yaml \
    --weights  yolov9-c-converted.pt \
    --imgsz    640 \
    --batch    4 \
    --epochs   50 \
    --close-mosaic 15 \
    --device   0 \
    --dtal \
    --reg_list -2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

### YOLOv9c-P2 (초소형 객체용)

논문에서는 더 높은 공간 해상도를 위해 YOLOv9c에 P2 feature layer를 추가한 변형도 평가합니다. config은 [models/detect/yolov9-c-p2.yaml](models/detect/yolov9-c-p2.yaml)에 포함되어 있으며, 동일한 방식으로 학습합니다.

### DSDL CLI 인자

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--dtal` | off | D-TAL(거리 기반 앵커 보충) 사용 |
| `--reg_list` | `0 1 … 15` | DFL bin 중심값. 음수를 넣으면 S-DFL, 소수를 넣으면 D-DFL이 됨 |

### τ 하이퍼파라미터

TAL의 목표 후보 개수 τ (논문 Algorithm 1)는 YOLOv9 기본 구현을 따라 `YOLOM` 환경변수로 조정합니다:

```bash
YOLOM=13 python train_dual.py --dtal --reg_list ...   # 논문 기본값 τ=13
YOLOM=4  python train_dual.py --dtal --reg_list ...   # ablation τ=4
YOLOM=20 python train_dual.py --dtal --reg_list ...   # ablation τ=20
```

### 멀티 GPU 학습

```bash
python -m torch.distributed.run --nproc_per_node 2 --master_port 9527 \
    train_dual.py \
    --cfg models/detect/yolov9-c.yaml \
    --data data/your.yaml \
    --device 0,1 --sync-bn \
    --dtal --reg_list ...
```

---

## 검증 (Validation)

학습된 체크포인트 검증:

```bash
python val_dual.py \
    --data    data/your_dataset.yaml \
    --weights runs/train/<실행명>/weights/best.pt \
    --imgsz   640 \
    --batch   1 \
    --conf    0.001 \
    --iou     0.7 \
    --device  0
```

mAP@50, mAP@50:95, 클래스별 지표가 출력됩니다. 검증 시점에는 별도의 DSDL 플래그가 필요하지 않습니다 — bin 정보가 체크포인트에 이미 저장되어 있습니다.

---

## 추론 (Inference / Detection)

이미지 또는 비디오에 대해 검출 실행:

```bash
python detect_dual.py \
    --source  path/to/images_or_video \
    --weights runs/train/<실행명>/weights/best.pt \
    --imgsz   640 \
    --conf    0.25 \
    --iou     0.45 \
    --device  0
```

유용한 옵션:
- `--save-txt` — 검출 결과를 YOLO 포맷 라벨 파일로 저장
- `--save-conf` — 라벨에 confidence 함께 저장
- `--view-img` — 실시간 미리보기 (데스크톱 환경)

---

## 논문 결과 재현

논문은 **YKH**(건설현장 PPE) 및 **VisDrone** 데이터셋에서 ablation 실험을 수행합니다. 특정 행을 재현하려면:

1. YOLOv9-c (또는 YOLOv9-c-P2)를 MS COCO 2017에서 YOLOv9 기본 설정으로 사전학습 (`--epochs 500 --close-mosaic 15 --batch-size 36`).
2. 목표 binning 설정과 τ로 50 epoch fine-tune:

```bash
# VisDrone @ 640px, τ=13, 전체 DSDL
YOLOM=13 python train_dual.py \
    --cfg     models/detect/yolov9-c.yaml \
    --data    data/visdrone.yaml \
    --weights runs/pretrain/yolov9-c/weights/last.pt \
    --imgsz 640 --batch 4 --epochs 50 --close-mosaic 15 \
    --dtal \
    --reg_list -2 -1.5 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.5 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```

[Binning 표](#binning-configurations)에 따라 `--reg_list`와 `--dtal`을 바꿔주면 각 ablation 행을 재현할 수 있습니다.

> **사전학습 가중치 관련.** `--reg_list`로 bin 개수를 바꾸면 detection head 채널 수(`4 * reg_max`)도 달라집니다. 백본은 기존 YOLOv9 체크포인트에서 그대로 전이되지만 detection head의 최종단의 node 갯수가 달라지며 이 부분만 새로 초기화됩니다. 이는 논문 3절에서 의도된 동작입니다.
다만 전체 모델 크기 대비 변경되는 파라메터 갯수가 극히 미미하기 때문에 training 혹은 inference 에 미치는 성능 영향은 없다고 봐도 무방합니다.

---

## 인용

연구에 DSDL을 사용하시면 다음과 같이 인용해주세요:

```bibtex
@article{KIM2026106929,
title = {Tiny object detection using Distance-guided, Signed, and Densified Learning (DSDL) for construction site safety monitoring},
journal = {Automation in Construction},
volume = {187},
pages = {106929},
year = {2026},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2026.106929},
url = {https://www.sciencedirect.com/science/article/pii/S0926580526001706},
author = {Seokhwan Kim and Taegeon Kim and Kichang Choi and Siheon Joo and Hongjo Kim},
keywords = {Tiny object detection, Model-agnostic enhancement, Distance-guided Task Alignment Learning, Signed Distribution Focal Loss, Densified Distribution Focal Loss},
abstract = {Ensuring construction safety requires accurate detection of personal protective equipment (PPE) to prevent major accidents. However, PPE items such as hooks and straps are typically extremely small (fewer than 162 pixels2), making them difficult to detect with conventional computer vision models. This paper identifies fundamental limitations in object detection, termed the Minnow Net Problem, in which tiny objects escape detection due to coarse anchor intervals, positive-only distribution bins, and low binning resolution. To address these challenges, this paper introduces Distance-guided Task Alignment Learning (D-TAL), Signed Distribution Focal Loss (S-DFL), and Densified Distribution Focal Loss (D-DFL) Learning (DSDL), a set of techniques that enhance tiny object detection without requiring modifications to model architectures. Experimental results demonstrate substantial improvements, achieving up to a 48.6 percentage-point gain in tiny object detection while preserving inference speed. DSDL functions as a model-agnostic enhancement applicable to modern one-stage object detection models, and the source code is publicly accessible.}
}
```

---

## 문의

본 구현 또는 논문 관련 문의사항은 아래로 연락 주시기 바랍니다:

- 김석환 — [yyksh2019@yonsei.ac.kr](mailto:yyksh2019@yonsei.ac.kr)
- 김홍조 (교신저자) — [hongjo@yonsei.ac.kr](mailto:hongjo@yonsei.ac.kr)

---

## Acknowledgements / 감사의 말

DSDL은 [YOLOv9 (Wang, Yeh, Liao, 2024)](https://github.com/WongKinYiu/yolov9) 위에 구현되었습니다. D-TAL 설계는 [TOOD (Feng et al., 2021)](https://arxiv.org/abs/2108.07755)의 Task Alignment Learning을 확장한 것이며, DFL 재정의는 [Generalized Focal Loss (Li et al., 2020)](https://arxiv.org/abs/2006.04388)에 기반합니다.
