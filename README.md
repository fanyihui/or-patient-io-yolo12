# 手术室患者入/出室时间检测（v1）

免训练验证：**YOLO-World + COCO 融合** + **伪床推断** + **ByteTrack**。面向真实入室场景：**带栏杆病床推进，患者盖被、通常只露出头部**。

> 推床在普通检测器里经常漏检。默认三层召回：① 开放词汇 `hospital bed`/`stretcher`；② COCO 家具辅检；③ 仍无床时用盖被头框扩成 `pseudo_bed`。

## 原理

1. 主检 YOLO-World + 辅检 YOLO12（bed/couch/dining table/person）
2. 仍无床框时：把盖被头扩成伪床，再与头配对
3. **病床**为事件主体；高大竖直 person 视为床旁医护，**默认不绘制、不触发事件**
4. **患者头只认躺着的头**（落在直立人体上半身的头/脸一律排除）
5. **病床推车 vs 器械推车**：器械车整体更小，不参与入出室；可用尺寸门槛 + 专用提示区分
6. **推器械车医护**：框变宽也不标成躺姿患者（直立宽框 + 靠近器械车排除）
7. 空床不触发；床+头关联成功后穿越门口 ROI → enter/exit
8. 兼容全身横向 / 合并框（合成视频与遮挡回退）

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

### Web 应用（推荐）：接入视频 → 截图标注门 ROI → 实时入室事件

```bash
pip install -r requirements.txt
python scripts/run_app.py --host 0.0.0.0 --port 8787
# 浏览器打开 http://127.0.0.1:8787/
```

流程：
1. 填写一路视频源（RTSP / HTTP / 摄像头 `0` / 文件路径）并连接  
2. **截图**冻结画面，点选**门外 ROI**与**门内 ROI**，保存坐标到 `configs/sites/`  
3. **开始监测**：识别推着病人的病床从门外进入门内，右侧实时显示入室时间  

事件与 ROI 会写入 `outputs/app_*`（`events.jsonl` / `events.csv` / `door_roi.yaml`）。

```bash
python scripts/generate_synthetic_video.py
python scripts/run_demo.py --source data/samples/or_door_synthetic.mp4 --output outputs/demo
# 强制 GPU：
python scripts/run_demo.py --source your.mp4 --output outputs/demo --device 0
```

### 流媒体实时识别与记录

```bash
# RTSP 摄像头 / NVR
python scripts/run_stream.py --source "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101" --device 0

# HTTP MJPEG / HLS
python scripts/run_stream.py --source "http://192.168.1.10/video.mjpg"
python scripts/run_stream.py --source "https://example.com/live/index.m3u8"

# 本地摄像头
python scripts/run_stream.py --source 0 --preview

# 降负载保实时（默认 target_fps=10），只记事件不写视频
python scripts/run_stream.py --source "rtsp://..." --target-fps 8 --no-video
```

输出目录（默认 `outputs/live_<时间>_<源>`）实时落盘：

| 文件 | 说明 |
|------|------|
| `events.jsonl` | 每条入/出室事件立即追加 |
| `events.csv` | 表格追加，便于 Excel |
| `events.json` | 汇总快照（持续刷新） |
| `annotated*.mp4` | 标注视频（可按 `video_segment_minutes` 切段） |
| `summary.json` | 结束时统计 |

断流默认自动重连；`Ctrl+C` 优雅退出并落盘。

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
stream:
  target_fps: 10
  drop_pending: true
  reconnect: true
output:
  save_events_jsonl: true
  save_events_csv: true
  video_segment_minutes: 30
target:
  mode: bed_patient
  stretcher:
    lying_aspect_wh: 1.45
    lying_max_height_ratio: 0.28
    reject_lying_near_equipment: true
    staff_push_max_aspect_wh: 1.35
    allow_merged_detection: true
zone:
  mode: roi
```

## 目录

```
configs/          # 默认配置
src/or_io/        # 检测流水线
scripts/          # 合成视频 / 离线推理 / 实时流 / 标定 / 探测
tests/            # 单元测试
```

## 远程仓库

https://github.com/fanyihui/or-patient-io-yolo12
