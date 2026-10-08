# 单目标视频追踪

本项目使用 YOLO 全帧检测与 Kalman 滤波，对离线视频中每次出现的单个目标建立轨迹。`--mode continuous-track` 可持续等待并跟踪先后出现的目标，直到视频结束。所有可读帧按顺序处理；`--detect-interval` 可以降低跟踪期间的检测频率，但不会跳过解码或逐帧记录。性能计时、ONNX/TensorRT 构建及 Jetson 部署见 [JETSON_PERFORMANCE.md](JETSON_PERFORMANCE.md)。

## 安装和运行

网络摄像机使用入口 `track_camera.py`，支持海康 RTSP/TCP、原始码流分段录像、完整分析与显式低延迟策略。`track_camera_batch.py` 可将一台摄像机的每张实时画面复制为 2/4/6/8 个逻辑摄像机输入，用固定 TensorRT batch 测试多路识别吞吐；它不模拟多路 RTSP/NVDEC。连接配置、静态检查、录像重放和当前验证范围见 [CAMERA_USAGE.md](CAMERA_USAGE.md)。

识别核心已与命令行、RTSP和文件输出解耦。外部系统可使用
`tracking.recognition.FramePacket`、`TrackingConfig` 和 `RecognitionEngine` 直接提交固定 batch；
`process_batch()` 返回按 `camera_id` 对应的结果，每个摄像机维护独立跟踪状态。模型应先调用
`warmup()`，完成后再启动外部采集。CSV/JSON只是可选的 `ResultSink`，不是核心接口。

```python
from tracking.recognition import FramePacket, RecognitionEngine, TrackingConfig

engine = RecognitionEngine(detector, TrackingConfig())
engine.warmup((640, 640, 3), count=5)
results = engine.process_batch(frame_packets)  # 数量必须匹配固定 TensorRT batch
engine.finish("system_stop")
```

在目标 Jetson（JetPack 6、Python 3.10）上安装并运行：

```bash
export UV_PROJECT_ENVIRONMENT=.venv-jetson
uv sync --python 3.10
uv run --no-sync python track_video.py /path/to/sample.mp4 --weights models/detect.pt
```

项目内的 `models/detect.pt` 已更新为本次重新训练的 **YOLO26s**，训练尺寸 **640×640**，类别为 `item`（ID 0）。运行不会重新训练。ONNX 导出与 TensorRT 构建需要额外依赖，安装方法见 [JETSON_PERFORMANCE.md](JETSON_PERFORMANCE.md#1-安装环境)。

默认使用 CPU、**640×640 方形推理输入**、0.25 置信度阈值和每帧检测；`--device 0` 使用 CUDA。
视频全图按比例缩放并补边至 640×640，输出视频及检测框仍使用原视频尺寸和坐标。
TensorRT 自动选择 `models/detect.640.fp16.engine`，必须由当前权重重新导出并构建。

旧 PT、ONNX 和三个 engine 已备份到 `models/archive/25fec41785b2/`。
`--imgsz 高 宽` 的矩形输入功能仍可显式使用，但之前 768×1280 的推荐和测速只适用于旧 YOLO26n；新模型默认采用训练尺寸 640×640。

首次检测到唯一候选才建立轨迹；无候选或有多个候选时保持 `WAITING`。超时丢失后，后续唯一检测会自动建立新轨迹段；默认 `track` 模式继续要求同类，`continuous-track` 模式为新段重新锁定类别。场景中有其他同类物体时，自动初始化无法保证目标身份。

```bash
uv run --no-sync python track_video.py /path/to/sample.mp4 --weights models/detect.pt --headless --output outputs/run_01
uv run --no-sync python track_video.py /path/to/sample.mp4 --weights models/detect.pt --headless --start-frame 230 --max-frames 100 --output outputs/run_02
uv run --no-sync python track_video.py /path/to/sample.mp4 --weights models/detect.pt --headless --detect-interval 3 --output outputs/run_03
```

## 持续单目标追踪

适用于画面中同时只有一个物体、不同物体先后出现的场景：

```bash
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python track_video.py \
  /path/to/sample.mp4 --weights models/detect.pt \
  --mode continuous-track --coast-seconds 0.5 \
  --backend tensorrt --device 0 --imgsz 640 \
  --headless --decoder gstreamer --encoder gstreamer
```

流程为 `WAITING → TRACKING → LOST_PENDING → LOST → WAITING`，循环至可读视频结束或 GUI 中按 Q/Esc 退出。短暂漏检后恢复仍属于同一段；按视频时间计算，距最后观测超过 `--coast-seconds` 后结束旧段，清空运动、尺寸及类别状态，下一帧开始搜索。显式 `--class-id` 仍限制所有目标的类别。等待期间每帧检测；跟踪期间按 `--detect-interval` 检测。

`--max-frames` 是整个运行的帧数上限，与目标消失无关。持续模式禁止同时传入该参数；若需短片段测试，可准备短视频或使用默认 `--mode track`。`--start-frame` 仍可用于选择分析起点。

新目标从 `segment=1` 开始依次编号。旧段结束后画面清除其轨迹，新目标以全新滤波状态启动，历史数据保留在文件中。等待期间 `segment` 和观测坐标为空；确认丢失的 `LOST` 帧保留刚结束的段号。若两次出现之间的间隔小于消失阈值，可能被归为同一段；这些编号表示轨迹段，不能证明物体身份。

`summary.json` 的 `segment_count` 和 `segments` 汇总每段信息：`start_frame/start_timestamp` 是首次接受观测，`last_observed_frame/last_observed_timestamp` 是最后接受观测，`end_frame/end_timestamp` 是确认丢失或运行结束时的帧与视频时间，`observed_frames` 仅统计接受观测的帧。`end_reason` 为 `tracking_timeout`、`end_of_readable_video` 或 `user_quit`。普通 `track` 模式也会输出这些统计，并可能以 `max_frames` 结束。

## 播放与进度

GUI 初始暂停：Space 播放或暂停，N 单步，Q 或 Esc 保存并退出。窗口默认按比例缩至 960×540 以内，`--window-size 800 450` 可调整预览上限。目标较晚出现时可用零起始的 `--start-frame`。输出目录必须不存在；未指定时创建带时间戳的新目录，输入视频始终只读。

终端在启动时打印视频尺寸和推理配置，默认约每处理一秒视频帧打印一次已处理帧数、模型检测帧数和耗时，结束时打印总帧数与速度。`--log-every N` 可改为每 N 帧打印，`--log-every 0` 关闭定期进度；最终统计仍会打印。详细配置和计时保存在输出目录中。

保存视频默认使用 `--encode-queue 0` 同步输出。指定的 4K 视频实测中，同步输出更快；异步输出的收益取决于视频尺寸和设备负载。
GStreamer 输出会尝试把管道容量扩大至 1 MiB，减少大图像的分块传输；系统不允许时使用实际可用容量，并在 `performance.json` 的 `environment.encoder_pipe_bytes` 中记录。
`--encode-queue 1` 或 `2` 可启用后台输出，最多保留对应数量的待写图像（另有 1 帧可能正在写入），满时等待，保持帧序、不丢帧；提交时复制图像，避免后续修改影响输出。
结束时会排空队列、等待编码器收尾并检查错误，因此最终 FPS 包含这部分时间。`--no-video` 时队列参数不生效。

## 匹配与丢失

处理流程为 YOLO 全帧检测 → 重叠框去重 → 类别和尺寸筛选 → 唯一候选校正 Kalman。`--nms-iou` 默认 0.5；`--class-id` 可按模型类别筛选。首次匹配后锁定类别；普通 `track` 模式在新段也保持类别锁定，持续模式在旧段结束后重置。前两次观测按减速模型估计速度；有多个候选时启用运动门限筛选，仍不能唯一确定就拒绝更新，不按最高分强行选择。

Kalman 预测采用指数减速：经过 `dt` 秒，速度乘以 `exp(-k·dt)`，其中 `k` 是 `--deceleration-rate`（默认 2.0 /秒，必须大于零）。预测速度逐渐趋近零，不会自行反向；`--acceleration-std` 控制过程噪声，观测仍可修正速度。减速率尚未用标注轨迹校准，应根据目标运动调整。

默认只有一个合格候选时，运动不一致会记录为 `detected_motion_disagreement`，但仍接受该观测；这也可能接受唯一误检。`--strict-motion-gate` 会要求唯一候选也通过运动门限。检测间隔内仅预测，实际检测未匹配则进入 `LOST_PENDING`。距离最后观测超过 `--coast-seconds`（默认 0.5 秒）后记录一帧 `LOST`，停止旧段预测；下一帧开始自动检测，找到唯一合格候选时建立新段。检测间隔不得大于允许的预测时长。当前不包含长时间遮挡后的身份重识别或相机运动补偿。

状态包括 `WAITING`（等待唯一检测）、`TRACKING`（有接受的观测）、`PREDICTED`（计划跳帧，仅预测）、`LOST_PENDING`（检测未匹配）和 `LOST`（旧段超时）。

## 输出

每次运行生成 `track.csv`、`track.jsonl`、`summary.json`、`timing.csv` 和 `performance.json`；默认还生成 `annotated.mp4`，`--no-video` 可关闭。绿色框和轨迹是接受的图像观测；青色是 Kalman 估计，橙色十字是本帧校正前预测。缺失观测的中心为空，不填零、不插值，自动建立新段时断开轨迹。预测轨迹不能当作遮挡期间的真实路径。

| 字段 | 含义 |
|---|---|
| `center_x/y`、JSON `center` | 接受的观测；跳帧和漏检时为空 |
| `estimated_x/y`、JSON `estimated_center` | 校正后估计或无观测时的纯预测 |
| `predicted_x/y` | 本帧校正前的运动预测 |
| `detection_ran` | 区分主动跳过检测与实际漏检 |
| `detection_confidence`、`class_id` | 接受检测的分数和类别；分数不是校准概率 |
| `seconds_since_observation` | 距离上次接受观测的秒数 |
| JSON `detections` | 本次通过阈值的所有候选，用于诊断匹配拒绝 |

`summary.json` 保存配置、视频信息和状态计数；状态计数不是准确率。导出视频按原视频标称 FPS 编码，不含音频。CSV/JSON 优先使用解码器时间戳，缺失或不递增时按 FPS 回退并记录来源。可变帧率视频应以 CSV/JSON 时间戳分析。解码器不再返回帧时记录 `end_of_readable_video`，目前无法自动区分正常结尾与文件损坏造成的提前结束。框中心也不等同于目标的物理质心。

## 验证

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

测试覆盖减速滤波、单目标匹配、丢失与自动重建、持续模式中的短暂漏检/空画面/下一目标、分段统计及完整视频输出、计时及读写流程。合成测试只验证软件流程，不代表真实录像精度；交互窗口需要人工验收。
