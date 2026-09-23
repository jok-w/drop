# Jetson 构建、计时与性能测试

本轮迁移了 `E:\detect_test` 的 NVDEC/NVENC、有限预读及模型指纹校验，当前项目独立运行，不需要旁边保留 detect_test。
参考硬件为 Orin NX 8GB、JetPack 6 / CUDA 12.6 / TensorRT 10。其他系统版本需先匹配 PyTorch 和 JetPack 依赖。

## 1. 安装环境

当前 Windows 的 `.venv` 保持原状；下面在 **目标 Jetson** 的项目目录运行，使用独立 `.venv-jetson`：

```bash
export UV_PROJECT_ENVIRONMENT=.venv-jetson
uv sync --python 3.10 --extra export
```

`pyproject.toml` 参考 detect_test 的安装源：Jetson torch 2.8.0 / torchvision 0.23.0 使用 jp6/cu126 wheel，Ultralytics 固定 8.4.158，NumPy 1.x。使用 opencv-python 4.11，并避免与其他 OpenCV 包混装。
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

本地已导出 `models/detect.onnx`，可以连同 `models/detect.pt` 一起复制到 Jetson。
本权重的架构是 **YOLO26n**。Ultralytics 8.4.158 中 `nms=None` 表示外部 NMS，`nms=False` 会选择无 NMS 检测头；导出和推理已统一外部 NMS，不能套用旧版本导出参数。

需要重新导出时：

```bash
uv run --no-sync python -m tracking.export_model pt-to-onnx \
  --model models/detect.pt --imgsz 1280 --output models/detect.onnx
```

在目标 Jetson 构建单个静态 FP16 engine：

```bash
uv run --no-sync python -m tracking.export_model onnx-to-engine \
  --onnx models/detect.onnx --imgsz 1280 --device 0 --workspace 2
```

默认输出 `models/detect.1280.fp16.engine`。固定 batch=1、输入 1×3×1280×1280；采用 FP16 层优化、FP32 输入输出。
workspace 单位 GiB，不是总内存上限；构建内存不足可用 `--workspace 1`。改变 imgsz 后要重新构建对应尺寸的 engine。
导出/构建先在临时文件上验证，成功后才替换输出；构建失败不覆盖原有有效 engine。
engine 带 Ultralytics 元数据头，不是直接供 trtexec 加载的裸 plan。
运行时校验 PT SHA256、输入尺寸、检测头策略、TensorRT/CUDA/GPU/Ultralytics 版本。强制 TensorRT 失败会报错，不会退回 PT。

## 3. 逐段计时

普通运行默认产生 `timing.csv` 与 `performance.json`，不必额外开启开关：

```bash
uv run --no-sync python track_video.py camera_20260910_144729.mp4 \
  --weights models/detect.pt --backend tensorrt --device 0 --imgsz 1280 \
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
| video_submit_ms | 提交编码器，不能解释为最终编码完成或持久落盘 |
| frame_wall_ms | 本帧取帧至结果提交；不含本行 timing.csv 写入与进度日志 |
| background_prepare_ms / sample_wait_ms / shared_copy_ms / color_convert_ms | 已消费当前帧对应的后台准备与硬件读取分项；不可与主线程耗时相加 |

未检测的帧，检测分项为空，不用 0 混入推理平均值。没有目标也会记录实际推理耗时。
首帧在初始化阶段读取并缓存：其 read_wait_ms=0 且 read_cached=True，真实读取耗时在 first_frame_read_ms 中（硬件 prime 首帧也可能计入 reader_open_ms）；其推理仍计入处理循环。

`performance.json` 包含每阶段 count/total/mean/P50/P95/max、初始化/预热/收尾、实际推理次数、运行参数、实际设备与后端、检测帧/预测帧/丢失帧分组。
导出后端会延迟初始化运行时，首次加载 ONNX session / TensorRT runtime 计入 warmup_ms；若 warmup=0，则计入首次实际 detector_ms。
`pipeline_fps_with_finalize` 包含线程结束、CSV 缓冲关闭及视频编码收尾；不包含模型加载、预热、起始帧定位。`application_wall_seconds` 另包含初始化，截止最终报告写出前。
这些保存时间不保证断电持久化，未每帧 fsync。

嵌套计时不能重复相加：preprocess/inference/postprocess/result_transfer 属于 detector_ms。后台读取与前台重叠，也不能与前台简单求和。
`--profile diagnostic` 会在检测调用边界额外同步 CUDA；`throughput` 不加这些额外同步，但保留 Ultralytics 自带的同步分项计时，因此是当前实现的吞吐测试，不是完全无 profiling 的纯 TensorRT 极限吞吐。

`--headless --no-video` 会完全跳过绘图。输出缩放不影响检测图像或 CSV 的原图坐标。
绘图默认保持原分辨率（`--output-max-width 0`），应把 1920 输出作为独立实验变量。

## 4. 检测器基准与结果一致性

普通轨迹流程进入 LOST 后会停止检测，不能拿整段平均 FPS 当检测速度。使用独立基准模式强制每帧检测，不进行跟踪、不受丢失状态影响：

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
  --weights models/detect.pt --candidate tensorrt --device 0 --imgsz 1280 \
  --start-frame 230 --max-frames 100 --warmup 10 --repeats 3 \
  --decoder opencv --report-dir outputs/benchmark_trt
```

输出 comparison.json、每次完整报告及日志。比较帧号/时间戳、同类框 IoU、中心差异和未匹配数量。
匹配采用同类别、IoU>=0.5 的贪心匹配，只衡量后端一致性，不是标注集准确率。
PT 与导出后端统一 rect=False 和 imgsz，默认 1280 正方形补边。不要将旧矩形输入的速度直接与这个新基线比较。

本地 CPU 对照可使用 `--candidate onnx --device cpu`。
基准模式无绘图/视频编码，但仍输出检测 JSON、CSV 与计时；完整输出速度请使用普通 track 模式另测。

## 5. 读取与保存对照

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

硬件集成测试覆盖真实 NVENC/NVDEC、非均匀 PTS、帧序和提前结束；Windows 会跳过它们。
本地已完成 ONNX 构建及 PT/ONNX 16 帧真实输入比较，接受的整像素框完全一致；这不是跨视频精度评估。
本地没有 CUDA/TensorRT，尚未生成目标 Jetson 的 engine，也尚未验证 Jetson 硬件性能。

可用 `python tools/package_jetson.py` 生成包含源码、测试、配置、PT 与 ONNX 的部署 ZIP；不包含视频、虚拟环境或历史输出。解压后按本文安装环境，在目标机生成 engine。
