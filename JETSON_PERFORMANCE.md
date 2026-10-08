# Jetson 构建、计时与性能测试

本轮迁移了 `E:\detect_test` 的 NVDEC/NVENC、有限预读及模型指纹校验，当前项目独立运行，不需要旁边保留 detect_test。
参考硬件为 Orin NX 8GB、JetPack 6 / CUDA 12.6 / TensorRT 10。其他系统版本需先匹配 PyTorch 和 JetPack 依赖。

## 1. 安装环境

下面在 **目标 Jetson** 的项目目录运行，使用 `.venv-jetson`：

```bash
export UV_PROJECT_ENVIRONMENT=.venv-jetson
uv sync --python 3.10 --extra export
```

`pyproject.toml` 是项目 Python 包的唯一依赖声明：Jetson torch 2.8.0 / torchvision 0.23.0 使用 jp6/cu126 wheel，Ultralytics 固定 8.4.158，NumPy 1.x。使用 opencv-python 4.11，并避免与其他 OpenCV 包混装。普通追踪只需 `uv sync --python 3.10`；本节使用 `--extra export` 安装 ONNX 导出与验证依赖。
TensorRT 与 GI 来自设备配套系统库，不从通用 PyPI 安装 TensorRT。先检查系统解释器：

```bash
/usr/bin/python3.10 -c 'import tensorrt; print(tensorrt.__version__)'
```

若系统缺少 bindings，使用匹配当前 JetPack 的 APT 源安装：

```bash
sudo apt install tensorrt python3-libnvinfer python3-libnvinfer-dev
```

系统可导入而虚拟环境不可导入时，为这个虚拟环境添加系统 bindings 路径：

```bash
TRT_SITE=$(/usr/bin/python3.10 -c 'import pathlib,tensorrt; print(pathlib.Path(tensorrt.__file__).resolve().parent.parent)')
uv run --no-sync python -c 'import pathlib,site,sys; assert sys.prefix != sys.base_prefix; pathlib.Path(site.getsitepackages()[0], "jetson-tensorrt.pth").write_text(sys.argv[1]+"\n")' "$TRT_SITE"
uv run --no-sync python -m tracking.check_environment --require-jetson
```

预检查输出 GPU、CUDA、TensorRT、Jetson 系统版本和编解码插件可用性。不能据此替代实际视频验收。
每次新终端运行 uv 前仍需设置 `UV_PROJECT_ENVIRONMENT=.venv-jetson`。

## 2. 模型构建

当前 `models/detect.pt` 为重新训练的 **YOLO26s**，训练尺寸 **640×640**，单类别 `item`。
检测、ONNX 导出和 engine 构建的默认输入统一为 640×640。更换权重后已重新导出 `models/detect.onnx`；旧文件不能用于新模型。
旧 PT、ONNX 和三个 engine 备份在 `models/archive/25fec41785b2/`。Ultralytics 8.4.158 中 `nms=None` 表示外部 NMS，`nms=False` 会选择无 NMS 检测头；导出和推理已统一外部 NMS，不能套用旧版本导出参数。

需要重新导出时：

```bash
uv run --no-sync python -m tracking.export_model pt-to-onnx \
  --model models/detect.pt --imgsz 640 --output models/detect.onnx
```

在目标 Jetson 构建单图静态 FP16 engine：

```bash
uv run --no-sync python -m tracking.export_model onnx-to-engine \
  --onnx models/detect.onnx --imgsz 640 --device 0 --workspace 2
```

默认输出 `models/detect.640.fp16.engine`。固定 batch=1、输入 1×3×640×640；采用 FP16 层优化、FP32 输入输出。
多逻辑摄像机测试可复用同一动态 ONNX 构建固定 batch engine：

```bash
uv run --no-sync python -m tracking.export_model onnx-to-engine \
  --onnx models/detect.onnx --imgsz 640 --batch 2 --device 0 --workspace 2
uv run --no-sync python -m tracking.export_model onnx-to-engine \
  --onnx models/detect.onnx --imgsz 640 --batch 4 --device 0 --workspace 2
uv run --no-sync python -m tracking.export_model onnx-to-engine \
  --onnx models/detect.onnx --imgsz 640 --batch 6 --device 0 --workspace 2
```

默认分别输出 `models/detect.640.b2.fp16.engine`、`detect.640.b4.fp16.engine` 和 `detect.640.b6.fp16.engine`。固定 batch engine 只接受完全匹配的图片数量，不会自动回退到单图。
workspace 单位 GiB，不是总内存上限；构建内存不足可用 `--workspace 1`。改变 imgsz 后要重新构建对应尺寸的 engine。
导出/构建先在临时文件上验证，成功后才替换输出；构建失败不覆盖原有有效 engine。
engine 带 Ultralytics 元数据头，不是直接供 trtexec 加载的裸 plan。
运行时校验 PT SHA256、输入尺寸、固定 batch、检测头策略、TensorRT/CUDA/GPU/Ultralytics 版本。强制 TensorRT 失败会报错，不会退回 PT。

### 历史记录：旧 YOLO26n 的矩形输入

以下矩形构建和结果属于备份的旧 YOLO26n，不作为新 YOLO26s 的运行配置。新模型采用前述 640×640 命令。
`--imgsz` 支持一个整数（正方形），或两个整数 **高 宽**；每个尺寸必须是 32 的倍数。
本次 1920×1080 视频测试推荐 `--imgsz 768 1280`：图像缩放到 1280×720，再上下各补 24 像素。
相对 1280×1280，目标缩放比例相同，输入像素减少 40%。
方形输入的顶部补边为 280 像素，改为 24 像素相差 256（32 的倍数），保留了模型步长 32 的网格对齐。
736×1280 仅上下各补 8 像素，像素减少 42.5%，但本次测试的检测差异更大，详见下方结果。
全图仍参与检测，输出框保持原视频坐标。其他宽高比的视频应另行选择尺寸并验证。

复用动态 ONNX，在目标设备构建：

```bash
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python -m tracking.export_model onnx-to-engine \
  --onnx models/archive/25fec41785b2/detect.onnx --imgsz 768 1280 --device 0 --workspace 1
```

默认输出 `models/archive/25fec41785b2/detect.768x1280.fp16.engine`，运行时使用相同的 `--imgsz 768 1280` 自动选择它。
历史复测时还需用 `--weights models/archive/25fec41785b2/detect.pt` 指向旧权重。

## 3. 逐段计时

普通运行默认产生 `timing.csv` 与 `performance.json`，不必额外开启开关：

```bash
uv run --no-sync python track_video.py camera_20260910_144729.mp4 \
  --weights models/detect.pt --backend tensorrt --device 0 --imgsz 640 \
  --headless --start-frame 230 --max-frames 100 --warmup 10 \
  --decoder gstreamer --decode-prefetch 2 --read-ahead 2 \
  --encoder gstreamer --video-bitrate 8000000 --output-max-width 1920 \
  --output outputs/jetson_full
```

`timing.csv` 每帧包含：

| 字段 | 含义 |
|---|---|
| read_wait_ms | 主线程获取当前帧的等待；不是纯磁盘或 NVDEC 解码时间 |
| detector_ms | 整次检测调用，包含预处理、模型、后处理与结果转换 |
| preprocess_ms | Ultralytics 预处理，包含其输入转换/上传部分 |
| inference_ms | Ultralytics 同步计时的模型执行，不等于纯 GPU kernel 时间 |
| postprocess_ms | Ultralytics 后处理，包括 NMS |
| result_transfer_ms | 检测结果转 CPU、列表与坐标校验 |
| detector_overhead_ms | 模型调用总时间中其他框架开销 |
| tracking_ms | 检测之外的关联、Kalman 与记录准备 |
| draw_ms | 输出缩放、绘框与轨迹 |
| display_ms | GUI 等待及人工操作；headless 为 0 |
| data_write_ms | track.csv / track.jsonl 序列化与缓冲写入 |
| video_submit_ms | 同步模式为编码器写入；异步模式为图像复制、等待队列及入队；均不是最终编码完成或持久落盘 |
| frame_wall_ms | 本帧取帧至结果提交；不含本行 timing.csv 写入与进度日志 |
| background_prepare_ms / sample_wait_ms / shared_copy_ms / color_convert_ms | 已消费当前帧对应的后台准备与硬件读取分项；不可与主线程耗时相加 |

未检测的帧，检测分项为空，不用 0 混入推理平均值。没有目标也会记录实际推理耗时。
首帧在初始化阶段读取并缓存：其 read_wait_ms=0 且 read_cached=True，真实读取耗时在 first_frame_read_ms 中（硬件 prime 首帧也可能计入 reader_open_ms）；其推理仍计入处理循环。

`performance.json` 包含每阶段 count/total/mean/P50/P95/max、初始化/预热/收尾、实际推理次数、运行参数、实际设备与后端、检测帧/预测帧/丢失帧分组。
导出后端会延迟初始化运行时，首次加载 ONNX session / TensorRT runtime 计入 warmup_ms；若 warmup=0，则计入首次实际 detector_ms。
`pipeline_fps_with_finalize` 包含线程结束、CSV 缓冲关闭及视频编码收尾；不包含模型加载、预热、起始帧定位。`application_wall_seconds` 另包含初始化，截止最终报告写出前。
默认 `--encode-queue 0` 同步输出；`1`、`2` 启用有限队列的后台视频输出，分别最多排队 1、2 帧。4K 实测未显示异步收益，因此采用同步默认值。
异步时 `performance.json` 的 `async_encoder` 记录提交/完成帧数、队列峰值和 `background_write_ms` 的统计。
后台写入与前台重叠，不能把它加到 `frame_wall_ms`；队列排空计入 `encoder_finalize_ms` 和总处理时间。
正常退出会确认全部提交帧写入并收尾；编码写入或收尾失败会报错，取消时由后台线程清理编码器。
这些保存时间不保证断电持久化，未每帧 fsync。

嵌套计时不能重复相加：preprocess/inference/postprocess/result_transfer 属于 detector_ms。后台读取与前台重叠，也不能与前台简单求和。
`--profile diagnostic` 会在检测调用边界额外同步 CUDA；`throughput` 不加这些额外同步，但保留 Ultralytics 自带的同步分项计时，因此是当前实现的吞吐测试，不是完全无 profiling 的纯 TensorRT 极限吞吐。

`--headless --no-video` 会完全跳过绘图。输出缩放不影响检测图像或 CSV 的原图坐标。
绘图默认保持原分辨率（`--output-max-width 0`），应把 1920 输出作为独立实验变量。

## 4. 检测器基准与结果一致性

普通轨迹流程可能按检测间隔跳过推理，确认 `LOST` 的一帧也不推理；随后会继续搜索下一目标。不能拿整段平均 FPS 当检测速度。使用独立基准模式强制每帧检测，不进行跟踪、不受丢失状态影响：

```bash
uv run --no-sync python track_video.py camera_20260910_144729.mp4 \
  --weights models/detect.pt --backend tensorrt --device 0 \
  --mode detect-benchmark --headless --no-video \
  --start-frame 230 --max-frames 100 --warmup 10 \
  --decoder opencv --output outputs/trt_detector
```

PT/TRT 顺序比较，每组进程独立，重复三次并交替执行顺序：

```bash
uv run --no-sync python -m tracking.benchmark camera_20260910_144729.mp4 \
  --weights models/detect.pt --candidate tensorrt --device 0 --imgsz 640 \
  --start-frame 230 --max-frames 100 --warmup 10 --repeats 3 \
  --decoder opencv --report-dir outputs/benchmark_trt
```

输出 comparison.json、每次完整报告及日志。比较帧号/时间戳、同类框 IoU、中心差异和未匹配数量。
匹配采用同类别、IoU>=0.5 的贪心匹配，只衡量后端一致性，不是标注集准确率。
PT 与导出后端统一 rect=False 和 imgsz，默认 640 正方形补边。比较后端时应保持尺寸一致；比较尺寸时应保持后端一致。

复现旧模型的正方形/矩形对比（帧号从 0 开始；180–479 共 300 帧，重复三次）：

```bash
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python -m tracking.benchmark \
  /home/mtr/Desktop/test_detect/camera_20260910_143934-1920x1080.mp4 \
  --weights models/archive/25fec41785b2/detect.pt --baseline tensorrt --candidate tensorrt --device 0 \
  --imgsz 1280 --candidate-imgsz 768 1280 \
  --start-frame 180 --max-frames 300 --warmup 10 --repeats 3 \
  --decoder gstreamer --report-dir outputs/benchmark_rect
```

`--imgsz` 指基线尺寸，`--candidate-imgsz` 指候选尺寸；不指定候选尺寸时沿用基线。
可分别用 `--baseline-engine` 和 `--engine` 指定两个 TensorRT 文件。
`comparison.json` 的 `runs[].role`、`median_detector_mean_ms` 以 baseline/candidate 区分实验组。
`baseline_only_frames` 表示基线有检测、候选无检测的帧，`candidate_only_frames` 相反；
`unmatched_baseline_boxes` / `unmatched_candidate_boxes` 还包含 IoU 未达阈值的框。
这些差异必须结合原图或人工标注复查，双方均未检出的目标不会被一致性比较发现。

### 历史实测：旧 YOLO26n 矩形输入（不适用于新权重）

视频 `camera_20260910_143934-1920x1080.mp4`，第 180–479 帧，每组 300 帧、三轮交替执行；
FP16 TensorRT 10.3、device 0、conf=0.25、NMS IoU=0.5、warmup=10、NVDEC、逐帧检测、无绘图和视频编码。
设备查询为 MAXN，未额外锁定频率。下表为每轮平均值的三轮中位数；两个实验各自有配对基线。

| 实验 | 方形检测 ms/帧 | 矩形检测 ms/帧 | 检测速度提升 | 方形有框、矩形无框 | 矩形有框、方形无框 |
|---|---:|---:|---:|---:|---:|
| 736×1280 | 64.73 | 57.96 | 11.7% | 7 帧 | 3 帧 |
| 768×1280 | 66.00 | 58.29 | 13.2% | 1 帧 | 0 帧 |

768 组的推理阶段从 26.33 ms 降至 20.98 ms；含数据输出及收尾、排除初始化的基准速度从 14.89 提升到 16.83 FPS。
预处理仍约 31.82 ms，是后续优化重点。检测总耗时减少 11.7%，其倒数表示的速度提升为 13.2%。
三轮检测结果差异一致：300 帧中方形有检出 78 帧，768 矩形 77 帧；唯一少检出的帧为 **436**（帧号从 0 开始）。
该帧方形置信度 0.2576，单帧低阈值诊断得到矩形置信度 0.2494、同样的整数框，略低于正式阈值 0.25。
正式对比和运行的阈值仍为 0.25。匹配框平均 IoU=0.9928，中心差异约 0.099 个原图像素。
用保存的检测结果回放持续跟踪，两者均得到 7 段轨迹；768 的其中一段从第 437 帧开始，比基线晚 1 帧。
736 方案则得到 8 段轨迹，因此本视频优先选择 768。没有人工标注，不能把这些一致性指标当作真实召回率。

原始报告分别为 `outputs/rect_optimization/benchmark/comparison.json` 和
`outputs/rect_optimization/benchmark_768/comparison.json`；跟踪回放脚本为 `outputs/rect_optimization/analyze.py`。

768 矩形另完成原视频全程 `continuous-track` 验收：798 帧输入/视频输出/JSON 行数一致，
1920×1080、约 25.035 FPS 的输出视频，11 段轨迹，正常读至 EOF。
包含绘图、硬件编码与收尾的处理时间 52.94 秒（15.07 FPS），包含初始化的应用时间 58.38 秒。
787 次模型调用，11 次确认丢失的帧按现有状态机跳过推理；不能用此吞吐量代替逐帧检测基准。
完整产物在 `outputs/rect_optimization/full_video_768/`。65 项回归测试通过，包含真实硬件编解码。

本地 CPU 对照可使用 `--candidate onnx --device cpu`。
基准模式无绘图/视频编码，但仍输出检测 JSON、CSV 与计时；完整输出速度请使用普通 track 模式另测。

## 5. 读取与保存对照

### 同步/异步输出对比

以下工具对同一 300 帧交替运行三轮，使用同一模型和输出尺寸；结束后完整解码每个视频，逐帧比较图像像素与轨迹记录：

```bash
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python -m tracking.benchmark_encoder \
  /home/mtr/Desktop/drop/camera_20260910_144729.mp4 \
  --weights models/detect.pt --backend tensorrt --device 0 --imgsz 640 \
  --decoder gstreamer --encoder gstreamer --encode-queue 2 --output-max-width 0 \
  --start-frame 180 --max-frames 300 --warmup 10 --repeats 3 \
  --report-dir outputs/benchmark_encoder_4k
```

使用 `track` 模式限制测试帧数；正式运行仍可使用 `continuous-track` 处理至 EOF。
工具输出每轮完整报告、视频和 `comparison.json`。吞吐比较包含队列及编码器收尾，初始化另计。
若视频帧数、尺寸、FPS 不符，或同步/异步的视频像素和轨迹有差异，工具会报错，不把它视为通过。

#### 当前指定的 4K 视频

`camera_20260910_144729.mp4`：HEVC、3840×2160、约 25.035 FPS、1644 帧。保持原始 4K 输出，使用新 YOLO26s 的 640×640 FP16 TensorRT engine，每帧检测（确认 LOST 的帧按原逻辑不检测）。测试第 180–479 帧，每种配置三轮，表内为中位数，包含编码收尾，不含模型加载、预热和起始帧定位。

| 编码配置 | 300 帧处理时间 | 处理吞吐 |
|---|---:|---:|
| 原 64 KiB 管道，同步 | 22.53 s | 13.32 FPS |
| 原 64 KiB 管道，异步队列 2 | 23.38 s | 12.83 FPS |
| 新 1 MiB 管道，同步（当前默认） | 21.57 s | 13.91 FPS |
| 新 1 MiB 管道，异步队列 2 | 22.15 s | 13.55 FPS |

扩大管道后同步吞吐实测提升约 4.5%；异步未显示稳定收益，默认改为 `--encode-queue 0`，仍可显式启用异步做对照。GStreamer 的读取分块同步调整至 1 MiB；若操作系统拒绝扩大管道，则按实际容量写入。实际容量保存在 `performance.json` 的 `environment.encoder_pipe_bytes` 中。

这些是同一设备上两个批次的片段测试，受设备负载和频率波动影响，不能视为所有视频的固定提速比例；当前 4K 保存速度尚未达到源视频约 25 FPS。此前的 1080p 提速结论不适用于此视频。

原始结果：`outputs/async_encoder_4k/benchmark/comparison.json`、`outputs/async_encoder_4k/buffer_1m/comparison.json`。跨版本第一组同步输出的 H.264 码流和轨迹文件完全一致：`outputs/async_encoder_4k/buffer_cross_validation.json`。扩大管道后的 74 项回归测试通过，含硬件编解码：`outputs/async_encoder_4k/tests_buffer.log`。
两批共 12 份输出均完整解码为 300 帧，各轮同步/异步的逐帧像素及轨迹记录完全一致。

正式处理完整视频（持续寻找下一目标，保留 4K 输出）：

```bash
cd /home/mtr/Desktop/drop
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python track_video.py \
  /home/mtr/Desktop/drop/camera_20260910_144729.mp4 \
  --weights models/detect.pt --backend tensorrt --device 0 --imgsz 640 \
  --mode continuous-track --headless \
  --decoder gstreamer --encoder gstreamer --encode-queue 0 \
  --output-max-width 0 --warmup 10
```

#### 历史 1080p 视频

历史 1080p 测试：新 YOLO26s / 640×640（`camera_20260910_143934-1920x1080.mp4`，1080p 输出，第 180–479 帧，三轮中位数）。此表不代表用户指定的 4K 视频：

| 指标 | 同步队列 0 | 异步队列 2 |
|---|---:|---:|
| 300 帧处理时间，含收尾 | 14.43 s | 11.76 s |
| 处理吞吐 | 20.79 FPS | 25.51 FPS |
| 应用时间，含加载/预热/定位 | 20.77 s | 18.08 s |
| 主线程视频提交平均耗时 | 11.28 ms/帧 | 1.33 ms/帧 |

吞吐提升约 22.7%，处理时间减少约 18.5%。后台写入本身仍有耗时，通过与检测重叠减少主线程等待。
六份视频均完整解码为 300 帧，三组同步/异步的逐帧像素及轨迹 JSON 完全一致；异步提交/完成均为 300 帧。
72 项回归测试全部通过，包含队列背压、复制隔离、取消、写入/收尾错误、超时及硬件编解码。
原始结果：`outputs/async_encoder/benchmark/comparison.json`，测试日志：`outputs/async_encoder/tests.log`。
这些速度来自该视频片段，不能保证所有视频都达到相同吞吐。

### 提前缩小 4K 视频的对照（2026-09-28）

将 `camera_20260910_144729.mp4` 用 FFmpeg bilinear + H.264 CRF 18 转成 640×360，保持 1644 帧；完整软件转码耗时 167.60 秒。两组使用同一 640×640 TensorRT engine、GStreamer 解码，并统一输出 640×360。当前设备内存条件下 NVENC/VIC 初始化失败，因此本次两组均使用 **OpenCV 编码**，不能与此前保留 4K、NVENC 输出的 FPS 直接对比。

第 180–479 帧、每组三轮交替测试的中位数：原 4K 输入 **17.61 FPS**，提前缩小 **25.14 FPS**，吞吐增加约 42.8%。该速度不含提前转码时间。

另做各 300 次逐帧检测，排除跟踪器跳过 LOST 帧的影响：有检测结果的帧数由 **55 降至 41**，14 帧仅原图检出，0 帧仅缩小图检出。此为一致性差异，没有人工真值，不能当作真实漏检率。六份输出均完整解码为 300 帧、640×360；转码的时间戳最大舍入误差为 20 ms，检测按帧索引对齐。

因此暂不建议直接用这份提前缩小的视频替换原输入。详细记录、复现脚本和检测差异见 `outputs/pre_scaled640/REPORT.md`、`comparison.json` 和 `detection_count_changes.csv`。下一步可以单独测试解码阶段缩放，省去二次有损编码，但仍需验证检测一致性。

### 读取/保存组合

保持模型、输入/输出尺寸、检测频率一致，每组使用新输出目录：

| 组 | decoder | read-ahead | encoder |
|---|---|---:|---|
| 软件读写基线 | opencv | 0（OpenCV 忽略预读） | opencv |
| 只改读取 | gstreamer | 0 | opencv |
| 增加一帧预读 | gstreamer | 1 | opencv |
| 增加两帧预读 | gstreamer | 2 | opencv |
| 再改编码 | gstreamer | 2 | gstreamer |

GStreamer appsink 的 decode-prefetch 固定为 2，另一个 read-ahead 控制完整 BGR 帧准备，两者不相同。
硬件读取保持输入原尺寸与 PTS。所有队列满时阻塞，不跳帧；4K BGR 两帧额外约 50MB，不含解码器内部参考帧和共享内存。
NVDEC 路径采用共享内存复制与 BGR 转换，不是零拷贝。
auto 模式只允许启动探测失败时回退；强制 gstreamer 失败直接报错，运行中错误不偷偷重新读视频。
比较读取速度以整体含收尾 FPS 为主，不要只看主线程等待下降。正式测速关闭 GUI，保持设备功耗模式、频率和温度可比，顺序执行至少三次。

## 6. 验证与当前状态

```bash
uv run --no-sync python -m unittest discover -s tests -v
RUN_JETSON_GSTREAMER_TEST=1 uv run --no-sync python -m unittest discover -s tests -p 'test_video_*.py' -v
```

硬件集成测试覆盖真实 NVENC/NVDEC、非均匀 PTS、帧序和提前结束；需要在目标 Jetson 上运行。
旧模型曾完成 ONNX、PT/ONNX 和三个尺寸的 TensorRT 验证，历史实测在 `outputs/rect_optimization/`。
新 YOLO26s 的导出、构建和验证记录在 `outputs/model_640/`；两次实验使用不同权重，性能和检出结果应分别解读。
新模型已生成 `models/detect.640.fp16.engine` 并通过 GPU 预热。
默认参数下，第 180–279 帧共 100 帧的 PT/TensorRT 对比中，两者均有 25 帧检出；整数框 IoU=1.0、检出数量差异为 0。
该对比只验证后端一致性，不代表标注集准确率。另用两次目标出现的 76 帧片段完成持续跟踪和硬件输出，
正常得到 2 段轨迹，输出视频及逐帧数据均为 76 帧、读至 EOF。62 项普通回归测试通过，3 项独立硬件测试未在本轮重跑。
详细验证数据见 `outputs/model_640/validation.json`。
engine 与 PT 指纹和设备运行库绑定，更换权重或运行环境后需重新构建。

可用 `python tools/package_jetson.py` 生成包含源码、测试、配置、PT 与 ONNX 的部署 ZIP；不包含视频、虚拟环境或历史输出。解压后按本文安装环境，在目标机生成 engine。
