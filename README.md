# 手术室患者入/出室时间检测（v1）

免训练验证：**YOLO-World 开放词汇** + **ByteTrack**。面向真实入室场景：**带栏杆病床推进，患者盖被、通常只露出头部**。

> 重要：普通 COCO `yolo12n` 对手术室推床 / 盖被只露头经常**检不出床和病人**（只标出站立医护）。默认已改为 YOLO-World，用文本提示检测 `hospital bed` / `stretcher` / `human head`。

## 原理

1. YOLO-World 按提示检测病床（hospital bed / stretcher / gurney）与人（person / human head / patient）
2. **病床**为事件主体；**患者证据**优先为「中心落在床内的小框」（头部）
3. 高大竖直 person 视为床旁医护，不单独触发
4. 空床不触发；床+头关联成功后穿越门口 ROI → enter/exit
5. 兼容全身横向 / 合并框（合成视频与遮挡回退）

```yaml
model:
  backend: world
  weights: yolov8s-worldv2.pt
  prompts: [person, human head, patient, hospital bed, stretcher, gurney, bed]
target:
  mode: bed_patient
  stretcher:
    patient_appearance: covered_head
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
