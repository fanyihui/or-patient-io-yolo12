# 手术室患者入/出室时间检测（v1）

免训练验证：**YOLO-World + COCO 融合** + **伪床推断** + **ByteTrack**。面向真实入室场景：**带栏杆病床推进，患者盖被、通常只露出头部**。

> 推床在普通检测器里经常漏检。默认三层召回：① 开放词汇 `hospital bed`/`stretcher`；② COCO 家具辅检；③ 仍无床时用盖被头框扩成 `pseudo_bed`。

## 原理

1. 主检 YOLO-World + 辅检 YOLO12（bed/couch/dining table/person）
2. 仍无床框时：把盖被头扩成伪床，再与头配对
3. **病床**为事件主体；高大竖直 person 视为床旁医护，**默认不绘制、不触发事件**
4. **患者头只认躺着的头**（落在直立人体上半身的头/脸一律排除）
5. **病床推车 vs 器械推车**：器械车整体更小，不参与入出室；可用尺寸门槛 + 专用提示区分
6. 空床不触发；床+头关联成功后穿越门口 ROI → enter/exit
7. 兼容全身横向 / 合并框（合成视频与遮挡回退）

```yaml
model:
  backend: fusion
  weights: yolov8s-worldv2.pt
  secondary_weights: yolo12n.pt
  equipment_prompts: [instrument cart, equipment cart, medical cart]
target:
  mode: bed_patient
  stretcher:
    allow_pseudo_bed: true
    patient_appearance: covered_head
    reject_upright_heads: true
    bed_min_area_ratio: 0.045   # 病床更大；更小/更方的当作器械车
    demote_bed_without_patient: true
output:
  hide_standing_staff: true
  hide_equipment_carts: true
```

## 快速开始

```bash
pip install -r requirements.txt
# 有 NVIDIA GPU 时请安装 CUDA 版 PyTorch，例如：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python scripts/generate_synthetic_video.py
python scripts/run_demo.py --source data/samples/or_door_synthetic.mp4 --output outputs/demo
# 强制 GPU：
python scripts/run_demo.py --source your.mp4 --output outputs/demo --device 0
```

排查漏检（打印每类检出次数，并可选保存叠加图）：

```bash
python scripts/probe_detections.py --source your.mp4 --max-frames 90 --save outputs/probe.jpg
```

默认 `model.device: auto`：检测到 CUDA 用 GPU 0，否则 CPU。

若必须回到 COCO YOLO12：

```yaml
model:
  backend: coco
  weights: yolo12n.pt
  classes: [0, 56, 57, 59, 60]
  conf: 0.12
```

## 配置要点

```yaml
target:
  mode: bed_patient
  stretcher:
    lying_aspect_wh: 1.10
    allow_merged_detection: true
zone:
  mode: roi
```

## 目录

```
configs/          # 默认配置
src/or_io/        # 检测流水线
scripts/          # 合成视频 / 推理 / 标定 / 探测
tests/            # 单元测试
```

## 远程仓库

https://github.com/fanyihui/or-patient-io-yolo12
