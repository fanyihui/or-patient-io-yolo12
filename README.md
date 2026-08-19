# 手术室患者入/出室时间检测（v1）

免训练验证：**YOLO12** + **ByteTrack**。面向真实入室场景：**带栏杆病床推进，患者盖被、通常只露出头部**。

## 原理

1. YOLO12 检测 `bed`（病床/栏杆床）与 `person`
2. **病床**为事件主体；**患者证据**优先为「中心落在床内的小框 person」（头部）
3. 高大竖直 person 视为床旁医护，不单独触发
4. 空床不触发；床+头关联成功后穿越门口 ROI → enter/exit
5. 兼容全身横向 / YOLO 合并框（合成视频与遮挡回退）

```yaml
target:
  mode: bed_patient
  stretcher:
    patient_appearance: covered_head   # 盖被只露头
    max_patient_to_bed_area: 0.55
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

默认 `model.device: auto`：检测到 CUDA 用 GPU 0，否则 CPU。启动时会打印 `[device] ...`。

## 配置要点

```yaml
target:
  mode: bed_patient
  stretcher:
    lying_aspect_wh: 1.20
    allow_merged_detection: true
zone:
  mode: roi
```

## 目录

```
configs/          # 默认配置
src/or_io/        # 检测流水线
scripts/          # 合成视频 / 推理 / 标定
tests/            # 单元测试
```

## 远程仓库

https://github.com/fanyihui/or-patient-io-yolo12
