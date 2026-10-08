# 网络摄像机运行说明

入口为 `track_camera.py`，复用当前 YOLO26s / 640×640 TensorRT FP16 engine。摄像机默认配置沿用 `cam_test_bash`：`192.168.1.64:554`、用户 `admin`、通道 1、主码流 101、RTSP/TCP。密码读取 `HIKVISION_PASSWORD` 或 `--password-file`，不会写入结果配置。

## 1. 不连接摄像机的静态检查

```bash
cd /home/mtr/Desktop/drop
export UV_PROJECT_ENVIRONMENT=.venv-jetson
uv run --no-sync python track_camera.py --check
```

检查系统 Python 的 GI/GStreamer、12 个所需插件、H.264/H.265 的两种模式管线在 NULL 状态下能否构造连接，以及 engine 的输入尺寸和当前权重指纹。不会执行 CUDA 推理，不证明摄像机在线或硬件在负载下可用。

## 2. 首次接入：完整分析

在 Bash 中输入密码，然后接收 60 秒作为首轮联调：

```bash
cd /home/mtr/Desktop/drop
export UV_PROJECT_ENVIRONMENT=.venv-jetson
read -rsp '摄像机密码: ' HIKVISION_PASSWORD
export HIKVISION_PASSWORD
uv run --no-sync python track_camera.py \
  --host 192.168.1.64 --rtsp-port 554 --channel 1 --stream main \
  --policy complete --segment-seconds 5 --seconds 60 \
  --weights models/detect.pt --backend tensorrt --device 0 --imgsz 640
```

也可以用 `--password-file /absolute/path/to/password`，文件只存密码，不是 Shell 脚本。`HIKVISION_HOST`、`HIKVISION_USER` 和 `HIKVISION_CHANNEL` 可覆盖默认连接配置。

`complete` 为默认模式：一个 RTSP 会话接收压缩码流，按关键帧封装成短 MPEG-TS；每个片段完成后通过现有 NVDEC 读取器逐帧分析。MPEG-TS 保留 H.264/H.265 的带内参数集，可兼容摄像机启动时更新 PPS 的情况。跨正常录像片段继续同一轨迹，模型只加载一次。后台接收继续录像，未分析片段进入 SQLite 队列；处理落后时打印待分析段数。

**此模式有分段等待延迟，默认约 5 秒起，实际取决于摄像机关键帧间隔及积压。** 它优先完整处理已保存的视频，不承诺实时低延迟。网络故障可能使源内容缺失，未完成录像文件不会冒充有效片段。

`--seconds` 限制接收进程的墙钟时长（包括连接/重连），停止接收后继续分析已完成片段，因此程序可能运行超过指定秒数。去掉 `--seconds` 即持续接收。Ctrl+C 停止本次运行，原始片段与未处理清单保留；它不承诺在中断时继续清空全部积压。

## 3. 显式选择低延迟模式

```bash
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python track_camera.py \
  --policy latest --seconds 60 \
  --weights models/detect.pt --backend tensorrt --device 0 --imgsz 640
```

此模式在同一个接收管线中将解析后的码流分成录像与 NVDEC 两路。原始压缩码流继续保存，检测从有限缓冲取较新的已解码帧；检测忙时会跳过旧帧。`skipped_before` 和汇总计数明确记录这些跳过，不能把它当成逐帧完整分析。

模型会在连接摄像机前使用与 `--imgsz` 相同尺寸的合成图像完成预热，避免首次真实帧触发 TensorRT 初始化并造成启动阶段集中跳帧。`--seconds` 从预热完成、开始接收摄像机之后计算；预热期间尚未连接摄像机，因此也不会生成录像。

可选 `--decode-width 1280`，在 VIC 侧按原比例缩放后再拷贝到 Python。仅用于 `latest`，默认 `0` 保留原尺寸；模型输入仍为 640×640。检测框和跟踪坐标映射回源视频坐标，结果记录 `source_size` 与 `processing_size`。缩放会改变检测结果，需先做一致性验证。

共享图像空间最多容纳一张 3840×2160 BGRx 图像；更大的实际处理图像会明确报错。进程间图像所有权通过确认消息交接，后台另保留至多一份最新样本，前台待处理队列为 1 帧。

### 3.1 单摄像机复制 batch 测试

`track_camera_batch.py` 只连接、解码并录像一次，将同一张实时 BGR 图像引用作为 N 个逻辑摄像机输入组成固定 batch，不再为每个槽位执行整帧 `copy()`。每个逻辑摄像机有独立的 `camera_id`、Kalman 状态和轨迹段；模型只加载、预热和调用一次。该入口仅支持 `latest`，用于测量 batch 推理、逐路跟踪和结果写入，**不代表 N 路 RTSP、NVDEC、网络和录像负载**。

先构建与测试数量完全匹配的 engine：

```bash
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python -m tracking.export_model onnx-to-engine \
  --onnx models/detect.onnx --imgsz 640 --batch 2 --device 0 --workspace 2
```

然后进行 60 秒真实画面测试：

```bash
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python track_camera_batch.py \
  --batch-size 2 --seconds 60 --weights models/detect.pt \
  --backend tensorrt --device 0 --imgsz 640
```

支持 batch 1/2/4/6/8，并自动选择 `models/detect.640.b2.fp16.engine`、`detect.640.b4.fp16.engine`、`detect.640.b6.fp16.engine` 等文件。`summary.json` 同时给出物理源帧数、逻辑帧数、总吞吐、源跳帧和各阶段分位数；`batch_timing.jsonl` 每个 batch 一行；`track.jsonl` 每个逻辑摄像机结果一行。`batch_detector_ms` 是整批耗时，`batch_assembly_ms` 是构造帧契约和逻辑摄像机列表的耗时，预处理/推理/后处理字段是每张图平均耗时。

## 4. 重新分析已保存录像

针对已停止采集的输出目录：

```bash
UV_PROJECT_ENVIRONMENT=.venv-jetson uv run --no-sync python track_camera.py \
  --replay /home/mtr/Desktop/drop/outputs/camera_实际批次目录 \
  --weights models/detect.pt --backend tensorrt --device 0 --imgsz 640
```

重放读取原清单内**全部已完成片段**，在新的自动生成目录重新分析；不会往旧轨迹追加重复记录，不修改旧清单。它不是从某一帧恢复模型状态的断点续跑。此功能可补算 `latest` 跳过的画面，或重新处理 Ctrl+C/分析失败后保留下来的片段。

## 5. 输出及统计

每次使用新的输出目录：

| 文件 | 内容 |
|---|---|
| `track.jsonl` / `track.csv` | 轨迹、源帧序号、PTS、会话编号、跳帧信息 |
| `summary.json` | 分析帧数、会话数、重连、轨迹段、待处理录像和错误 |
| `connection.jsonl` | 连接、视频格式、片段完成、重连、未正常收尾文件等事件 |
| `fragments.sqlite3` | 已完成片段的持久索引、时间范围及分析完成状态 |
| `session_*/raw_*.ts`（`complete`）/ `raw_*.mp4`（`latest`） | 原始压缩码流录像，未重新编码；两者均可用于 `--replay` |
| `session_*/events.jsonl` / `worker.log` | 每次连接的诊断信息 |

目前摄像机入口为无窗口处理，不生成带框预览视频；需要检查画面时可用现有离线入口分析录好的 MP4。该入口不调用 NVENC，录像只做码流封装。

- 每次重连分配新会话，新会话不沿用旧 Kalman 状态或目标身份；正常片段边界保持连续跟踪。
- `timestamp` 为会话中的源 PTS 时间，可能在下一会话重新开始，须结合 `session_id` 使用。
- `frame_ready_to_result_ms` 是可用图像到结果的本机延迟，不是摄像机曝光至结果的延迟。完整模式另记 `archive_wait_ms`；重放不报告旧运行的排队延迟。
- `wall_fps` 包含等待输入/录像片段、预热及收尾，不能当作纯推理速度。
- `skipped_before_processed_frames` 统计已处理帧前观察到的序号间隙；低延迟模式的最终 `unprocessed_decoded_frames` 仅在所有 worker 正常报告解码总数后提供，否则为 null。
- 单次超时默认 5 秒，重连退避上限 30 秒；不会通过增加无限内存队列掩盖积压。
- 原始录像不会循环删除。默认可用空间不足 512 MiB 时停止接收并报告错误；可用 `--min-free-mb` 配置阈值。磁盘异常可能影响同一管线的两个分支，不承诺失败时还能完整保存。

## 6. 当前验证范围

2026-09-30 已完成 USB 网卡直连、真实 RTSP/TCP 认证取流、H.265 录像、NVDEC 和 TensorRT 联调。摄像机主码流已从 3840×2160 调整为 2560×1440、25 fps。60 秒 `latest` 实测 1493 帧中分析 1474 帧、无重连；60 秒 `complete` 复测收到并分析 1337 帧、跳帧 0、待处理片段 0。该次 `complete` 在启动时出现一次超时重连，之后同一会话连续处理 1335 帧。20 秒 B2 复制测试完成 375 次 batch、750 个逻辑帧，总吞吐 37.15 fps；整批检测均值 46.09 ms、整批含复制/跟踪/写出均值 51.53 ms。15 秒 B4 测试完成 181 次 batch、724 个逻辑帧，总吞吐 47.54 fps；整批检测均值 69.82 ms、完整整批均值 80.11 ms。两次 batch 测试均无重连或分析错误。断网恢复和至少 10 分钟持续运行仍需单独验证。

2026-10-08 使用已保存的原生 2560×1440 画面完成 B6 稳态离线测试：300 批、1800 张图用时 32.32 秒，吞吐 9.28 batch/s、55.69 image/s；整批检测均值 107.50 ms、P95 117.27 ms。当时摄像机 USB 网卡为 `NO-CARRIER`，所以该结果不含实时 RTSP/NVDEC、跟踪和结果写出，不能冒充实时端到端测试。

`tests/rtsp_fixture.py` 提供本地 RTSP/TCP 回归测试工具，真实摄像机结果保存在各次输出目录的 `summary.json` 与 `connection.jsonl`。

接入后建议依次检查：60 秒完整模式、60 秒低延迟模式、断网恢复、至少 10 分钟运行；最后重放同一录像，对照分析数量和轨迹连续性。

## 7. USB 网卡直连摄像机的网络检查

本机摄像机 USB 网卡为 `enx3cab728438c4`。物理链路显示 `carrier on` 仅代表链路建立，不代表已获取 IP 或摄像机可以访问。若摄像机仍使用 `192.168.1.64`，而 USB 网卡没有 IPv4 地址，可在本机终端进行临时配置（需要本机 sudo 权限）：

```bash
sudo ip address add 192.168.1.205/32 dev enx3cab728438c4
sudo ip route replace 192.168.1.64/32 dev enx3cab728438c4 src 192.168.1.205
ip route get 192.168.1.64
```

该方案假设专用链路上 `192.168.1.205` 空闲。`/32` 路由仅将摄像机地址导向 USB 网卡，避免与本机其他 `192.168.1.x` 网卡的整段路由竞争。预期路由输出含 `dev enx3cab728438c4 src 192.168.1.205`。配置可能在重启、拔插或 NetworkManager 重新配置网卡后失效；验证连接成功后再决定持久配置。重复添加已有地址会提示 `File exists`，应先查看 `ip -4 address show dev enx3cab728438c4`。

如需撤销本次临时配置：

```bash
sudo ip route del 192.168.1.64/32 dev enx3cab728438c4
sudo ip address del 192.168.1.205/32 dev enx3cab728438c4
```
