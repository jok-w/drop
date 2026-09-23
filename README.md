# 单目标视频追踪

本项目使用 YOLO 全帧检测与 Kalman 滤波，对离线视频中的一个目标建立轨迹。所有可读帧按顺序处理；`--detect-interval` 可以降低检测频率，但不会跳过解码或逐帧记录。性能计时、ONNX/TensorRT 构建及 Jetson 部署见 [JETSON_PERFORMANCE.md](JETSON_PERFORMANCE.md)。

## 安装和运行

在目标 Jetson（JetPack 6、Python 3.10）上安装并运行：

```bash
export UV_PROJECT_ENVIRONMENT=.venv-jetson
uv sync --python 3.10
uv run --no-sync python track_video.py /path/to/sample.mp4 --weights models/detect.pt
```

人工框选需要桌面图形会话。项目内的 `models/detect.pt` 是已有权重，运行不会重新训练。ONNX 导出与 TensorRT 构建需要额外依赖，安装方法见 [JETSON_PERFORMANCE.md](JETSON_PERFORMANCE.md#1-安装环境)。

默认使用 CPU、1280 像素方形推理输入、0.25 置信度阈值和每帧检测。可用 `--imgsz 640` 减少计算量，但需检查小目标召回；`--device 0` 需要可用的 CUDA 环境。这些参数尚未经过独立评估集校准。

不提供初始 ROI 时，首次检测到唯一候选才建立轨迹；无候选或有多个候选时保持 `WAITING`。场景中有其他同类物体时，可用 `--roi X Y W H` 指定首帧目标，或在播放窗口按 R 框选。ROI 使用原视频像素坐标。唯一检测也可能是误检，自动初始化不保证目标身份。

```bash
uv run --no-sync python track_video.py /path/to/sample.mp4 --weights models/detect.pt --headless --roi 100 200 32 32 --output outputs/run_01
uv run --no-sync python track_video.py /path/to/sample.mp4 --weights models/detect.pt --headless --start-frame 230 --max-frames 100 --output outputs/run_02
uv run --no-sync python track_video.py /path/to/sample.mp4 --weights models/detect.pt --headless --detect-interval 3 --output outputs/run_03
```

GUI 初始暂停：Space 播放或暂停，N 单步，R 在当前帧重建轨迹段，Q 或 Esc 保存并退出。窗口默认按比例缩至 960×540 以内，`--window-size 800 450` 可调整预览上限；框选坐标会换算回原视频坐标。目标较晚出现时可用零起始的 `--start-frame`。输出目录必须不存在；未指定时创建带时间戳的新目录，输入视频始终只读。

终端在启动时打印视频尺寸和推理配置，默认约每处理一秒视频帧打印一次已处理帧数、模型检测帧数和耗时，结束时打印总帧数与速度。`--log-every N` 可改为每 N 帧打印，`--log-every 0` 关闭定期进度；最终统计仍会打印。详细配置和计时保存在输出目录中。

## 匹配与丢失

处理流程为 YOLO 全帧检测 → 重叠框去重 → 类别和尺寸筛选 → 唯一候选校正 Kalman。`--nms-iou` 默认 0.5；`--class-id` 可按模型类别筛选。首次匹配后锁定类别。前两次观测用于估计初速度；有多个候选时启用运动门限筛选，仍不能唯一确定就拒绝更新，不按最高分强行选择。

默认只有一个合格候选时，运动不一致会记录为 `detected_motion_disagreement`，但仍接受该观测；这也可能接受唯一误检。`--strict-motion-gate` 会要求唯一候选也通过运动门限。检测间隔内仅预测，实际检测未匹配则进入 `LOST_PENDING`。距离最后观测超过 `--coast-seconds`（默认 0.5 秒）后进入 `LOST`，停止预测和自动匹配；GUI 可按 R 人工建立新轨迹段，无窗口模式持续记录缺失。检测间隔不得大于允许的预测时长。当前不包含长时间遮挡后的身份重识别或相机运动补偿。

状态包括 `WAITING`（尚未初始化）、`TRACKING`（有接受的观测）、`PREDICTED`（计划跳帧，仅预测）、`LOST_PENDING`（检测未匹配）和 `LOST`（超时，需人工重新初始化）。

## 输出

每次运行生成 `track.csv`、`track.jsonl`、`summary.json`、`timing.csv` 和 `performance.json`；默认还生成 `annotated.mp4`，`--no-video` 可关闭。绿色框和轨迹是接受的图像观测；青色是 Kalman 估计，橙色十字是本帧校正前预测。缺失观测的中心为空，不填零、不插值，丢失或人工重建时断开轨迹。预测轨迹不能当作遮挡期间的真实路径。

| 字段 | 含义 |
|---|---|
| `center_x/y`、JSON `center` | 接受的观测；跳帧和漏检时为空 |
| `estimated_x/y`、JSON `estimated_center` | 校正后估计或无观测时的纯预测 |
| `predicted_x/y` | 本帧校正前的运动预测 |
| `detection_ran` | 区分主动跳过检测与实际漏检 |
| `detection_confidence`、`class_id` | 接受检测的分数和类别；分数不是校准概率 |
| `seconds_since_observation` | 距离上次接受观测的秒数 |
| JSON `detections` | 本次通过阈值的所有候选，用于诊断匹配拒绝 |

`summary.json` 保存配置、视频信息、状态计数和人工干预次数；状态计数不是准确率。导出视频按原视频标称 FPS 编码，不含音频。CSV/JSON 优先使用解码器时间戳，缺失或不递增时按 FPS 回退并记录来源。可变帧率视频应以 CSV/JSON 时间戳分析。解码器不再返回帧时记录 `end_of_readable_video`，目前无法自动区分正常结尾与文件损坏造成的提前结束。框中心也不等同于目标的物理质心。

## 验证

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

测试覆盖滤波、单目标匹配、丢失与人工重建、计时及读写流程。合成测试只验证软件流程，不代表真实录像精度；交互窗口需要人工验收。
