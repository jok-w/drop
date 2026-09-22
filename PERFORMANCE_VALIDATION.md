# 本地验证记录（2026-09-22）

## 已完成

- 增加逐帧 timing.csv 与 performance.json，含读取、模型分项、跟踪、绘图、数据保存、视频提交、初始化/预热/收尾。
- 独立 detect-benchmark 每帧推理，避免轨迹丢失后停推理导致平均 FPS 虚高。
- TensorRT 10 构建与元数据校验代码，静态 batch=1、FP16 层优化、默认 1280 输入。
- 从参考项目迁移 Jetson 硬件读写、PTS、有限预读、错误传播和收尾。
- 保留 CSRT；无窗口且无视频保存时不绘图。
- 提供 Jetson 环境配置、预检查、构建说明和独立进程对照脚本。

## 实际执行结果

本机：Windows 11、Python 3.13.12、torch 2.14.0+cpu、Ultralytics 8.4.158，CUDA 不可用。

1. `python -m unittest discover -s tests`：58 项，55 项通过，3 项 Jetson 硬件集成测试跳过。
2. 从提供的 PT 构建 `models/detect.onnx` 成功，ONNX 图检查与 CPU 输入验证通过；IR=8、opset=17，动态输入 NCHW。
3. 权重架构确认是 YOLO26n；修正 nms=False 会切换检测头的问题，统一 nms=None 外部 NMS。
4. 第 230～245 帧 PT/ONNX 比较：16 帧全部各匹配一个检测框，整像素框平均 IoU=1.0、中心差异=0、未匹配数量=0。这仅验证该片段的后端一致性。
5. 16 帧轨迹输出：6 次检测、10 帧预测；视频、轨迹、计时记录均为 16 帧。视频为 1280×720，CSV 保持原 3840×2160 坐标；未检测帧推理耗时为空。
6. 新导出后端避免为读取类别名重复加载模型，ONNX 单次加载及类别过滤实测通过。
7. 本地 TensorRT 构建预检失败：缺少 tensorrt，且没有 CUDA GPU；未生成 engine。不是已在 Jetson 验证的构建结果。

可查看：

- `outputs/perf_onnx_verified/comparison.json`：PT/ONNX 对照。
- `outputs/perf_track_output/performance.json`：完整轨迹流程分项。
- `outputs/perf_track_output/timing.csv`：逐帧计时。
- `outputs/perf_track_output/annotated.mp4`：缩放输出视频。
- `outputs/performance_tests_final.log`：单元与集成测试输出。

## 目标机待验证

Jetson 依赖安装、TensorRT 实际引擎构建与 FP16 一致性、NVDEC/NVENC 真机管道、预读带来的整体收益尚未验证。
按 JETSON_PERFORMANCE.md 在目标 Jetson 安装、构建并运行，不能把上述 CPU 耗时当作 Jetson 性能。
