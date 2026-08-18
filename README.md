# 手术室患者入/出室时间检测（v1）

免训练验证：**YOLO12** + **ByteTrack**，以 **病床 + 平躺患者** 关联后穿越门口 ROI，记录入/出室时间。

## 原理

1. YOLO12 检测 `bed`/`couch` 与 `person`
2. 区分 **平躺患者**（横向大框）与直立人员
3. **病床 ↔ 平躺患者** 空间关联
4. 仅关联成功才触发事件；空床、直立医护不计
5. YOLO 合成一框时可用 `allow_merged_detection` 回退
6. 组合中心穿越 ROI：`outside→inside` = enter

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
