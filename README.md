# 单目标视频追踪

性能计时、ONNX/TensorRT 构建、Jetson 硬件读写与测试命令见 [JETSON_PERFORMANCE.md](JETSON_PERFORMANCE.md)。
运行默认新增 `timing.csv` 和 `performance.json`。当前推理统一 `rect=False` 正方形补边，便于 PT/ONNX/TensorRT 公平对照。

## YOLO + Kalman：惰性模型实验的新入口

已经接入用户提供的 `detect.pt`，项目内副本为 `models/detect.pt`。不重新训练权重。
使用 `--weights` 明确启用 YOLO 后端，不传该参数仍运行下文的旧版流程。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv\Scripts\python.exe track_video.py camera_20260910_144729.mp4 --weights models/detect.pt
```

默认 CPU、推理输入长边 1280、置信度阈值 0.25、每帧检测。这些是工程起点，未经过独立评估集校准。
GUI 初始暂停，Space 播放，N 单步，R 重新框选目标，Q 退出。
未提供 ROI 时，首次检测到唯一候选就建立轨迹；没有候选或出现多个候选时等待。
若场景中存在其他同类物体，建议用 `--roi X Y W H` 指定初始目标，或按 R 框选。
唯一候选也可能是误检，自动初始化不构成身份保证。

无窗口处理与低频检测示例（检测间隔单位是视频帧数）：

```powershell
.\.venv\Scripts\python.exe track_video.py camera_20260910_144729.mp4 --weights models/detect.pt --headless --start-frame 230 --max-frames 100 --output outputs/yolo_example
.\.venv\Scripts\python.exe track_video.py camera_20260910_144729.mp4 --weights models/detect.pt --headless --detect-interval 3 --output outputs/yolo_interval3
```

可用 `--imgsz 640` 减少推理计算量，但需检查小目标召回损失；`--device 0` 需要可用的 CUDA 版 PyTorch 与显卡。
`--class-id` 按模型类别 ID 筛选；类别名称、权重 SHA256、推理参数和库版本写入 `summary.json`。
实际检测频率约为视频 FPS / detect-interval，仍解码和记录每一帧。

流程为 YOLO 全帧检测 → 重叠框去重 → 类别/尺寸筛选 → 唯一候选校正 Kalman。
`--nms-iou` 默认 0.5，控制检测器对重叠候选的抑制。
前两次唯一观测用于估计初速度，第二次观测尚不使用运动门限，之后启用运动一致性检查。
默认只有一个类别/尺寸合格候选时，运动不一致仅记录 `detected_motion_disagreement`，仍保留检测观测并校正滤波器。
这适合先独立评估检测器与滤波器，避免未经校准的运动模型吞掉正确检测；同时也可能接受唯一误检。
有多个候选时尝试运动门限筛选，仍不能唯一确定则拒绝。使用 `--strict-motion-gate` 可让唯一候选也必须通过门限。
因此起步阶段仍可能关联到唯一的同类误检，需要通过独立数据评估；多个候选不会强行匹配。
首次匹配后锁定类别。多个候选同时通过筛选时拒绝更新，不按最高分强行选择。
检测间隔内仅预测，实际检测却未匹配时进入短时丢失；距离最后观测超过 `--coast-seconds`（默认 0.5 秒）后停止预测与匹配，按 R 建立新轨迹段。
检测间隔不能大于允许的预测时长。当前不包含跨长遮挡的身份重识别，也没有相机运动补偿。

输出仍为 `annotated.mp4`、`track.csv`、`track.jsonl`、`summary.json`：

| 字段 / 画面 | 含义 |
|---|---|
| `center_x/y`、JSON `center`、绿色轨迹 | 接受的图像观测（初始化也可能来自人工），跳帧/漏检时为空 |
| `estimated_x/y`、JSON `estimated_center`、青色轨迹 | 校正后的 Kalman 估计或无观测时的纯预测；不是实测 |
| `predicted_x/y`、橙色十字 | 本帧校正前的运动预测 |
| `detection_ran` | 区分主动跳过检测与实际漏检 |
| `detection_confidence`、`class_id` | 接受的检测分数及类别；分数不是校准的正确概率 |
| `seconds_since_observation` | 距离上次接受观测的秒数 |
| JSON `detections` | 本次所有通过检测器阈值的候选，用于诊断关联拒绝 |

状态包括 WAITING（尚未初始化）、TRACKING（当前有观测）、PREDICTED（计划跳帧，仅预测）、LOST_PENDING（检测未匹配）、LOST（超时，需要人工重新初始化）。
青色轨迹在缺少观测时仍可能连续，不能据此推断反弹、遮挡期间的真实路径；丢失或人工重建时断开。
框中心受目标旋转与检测框变化影响，不等同物理质心。现有旧视频已用于开发，不能用其表现证明泛化能力。

## 旧版 CSRT / 局部亮度流程

对现实录像中的一个仿真模型做离线二维追踪。人工指定初始目标，默认先判断目标与背景能否通过局部亮度差分离：可以时采用局部外观定位，不可以时使用 CSRT。Kalman 提供搜索位置与运动一致性检查。所有解码帧按顺序处理，不为了实时播放而主动丢帧。

## 安装和启动（Windows PowerShell）

已在本项目 `.venv` 中配置运行环境，可以直接使用下方运行命令。在其他电脑重新安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

请在独立环境中安装 `opencv-contrib-python`，避免与其他 OpenCV 包混装。需要带桌面 GUI 的版本，不能用 headless 包进行人工框选。

```powershell
.\.venv\Scripts\python.exe track_video.py "E:\videos\sample.mp4"
```

选择首个目标框，按 Enter 确认。播放窗口初始暂停：

播放和框选窗口默认按比例缩小到 960×540 像素以内，也可以拖动窗口边缘调整。需要更小的窗口时，在启动命令后加 `--window-size 800 450`。框选坐标会自动转换回原视频坐标，追踪和导出仍使用原分辨率。

| 操作 | 按键 |
|---|---|
| 播放 / 暂停 | Space |
| 提交当前帧并显示下一帧 | N |
| 在当前帧重新框选并重置追踪 | R |
| 保存当前结果并退出 | Q / Esc |

目标较晚出现时，以零起始帧号指定初始化位置：

```powershell
.\.venv\Scripts\python.exe track_video.py "E:\videos\sample.mp4" --start-frame 125
```

无需窗口的可复现处理，ROI 为原视频像素坐标 `左上角x 左上角y 宽 高`：

```powershell
.\.venv\Scripts\python.exe track_video.py "E:\videos\sample.mp4" --headless --roi 100 200 32 32 --output outputs\run_01
```

输出目录必须不存在，避免覆盖之前的结果。不指定时创建带时间戳的新目录。输入视频始终只读。

## 数据和画面

- `annotated.mp4`：绿色框和线为接受的图像估计；橙色十字为校正前运动预测。缺失帧和人工重新初始化处断开轨迹。
- `track.csv`：逐帧时间、状态、接受的位置、预测位置、候选框、平方马氏距离、拒绝原因；缺失观测为空，不填零、不插值。
- `track.jsonl`：同样的逐帧记录，并包含预测测量空间的协方差矩阵，便于诊断门限。
- `summary.json`：配置、视频信息、状态计数、人工干预次数和运行耗时。状态计数不是准确率；GUI 模式耗时包含等待用户操作。

导出视频按原视频标称 FPS 编码，不包含音频；可变帧率视频应以 CSV/JSON 时间戳为分析依据。优先使用解码器报告的时间戳，缺失或不递增时使用 FPS 回退，并记录来源。解码器停止返回帧时记录 `end_of_readable_video`；目前无法自动区分正常结尾和损坏文件导致的提前终止。

## 追踪与丢失逻辑

默认 `--appearance auto` 会从初始框内外学习目标亮度、背景亮度、目标面积与对比度，启用条件包括明显的亮度差及相对紧凑的前景。适合当前录像中“深色模型、较亮地面”的场景，也支持较暗背景上的亮目标；不是通用的语义目标检测器。

启用后，逐帧在预测位置和最后观测附近搜索前景连通区域，检查面积变化、紧凑程度、亮度一致性、局部对比度和候选歧义。可靠且唯一的图像候选与运动预测冲突时，记录 `motion_reset` 并用图像测量重建运动状态，以适应旋转、反弹和镜头移动。短时未找到目标时保留空观测并继续局部搜索，重新定位时要求更强的对比度证据；超过 `--coast-seconds` 后停止自动搜索。没有独立的相机稳像模块，不能保证大幅镜头运动下继续追踪。

局部外观来源在输出中标为 `local_contrast`；`appearance_cost` 是用于候选排序的代价（越低越好），不是概率。`summary.json` 的 `appearance_active` 表示该次初始化是否满足启用条件。相近颜色、阴影、细线与目标粘连、复杂枝叶或过小目标仍可能造成漏检或误匹配。初始框应围住单个模型并尽量少包含背景。

使用 `--appearance off` 可以明确选择原有 CSRT 基线，或者在初始前景不可分离时自动使用以下流程：

1. 人工框选建立一个轨迹段，状态为 `TRACKING`。
2. 每帧按实际时间间隔预测 `[cx, cy, vx, vy]`，CSRT 独立给出候选框。
3. 检查框是否有效、尺寸是否异常突变，并计算 `d² = rᵀ S⁻¹ r`，其中 `S = H P预测 Hᵀ + R`。
4. 通过门限才用候选中心校正 Kalman。CSRT 的成功标志不等于已确认真实目标。
5. 任一检查失败，丢弃可能已被错误画面污染的 CSRT，进入 `LOST_PENDING`，该帧接受的位置为空。如果仅因运动门限被拒绝，在 `--coast-seconds` 时间内，用最后可靠画面和框重建 CSRT，尝试定位后续帧；通过检查才恢复，并记录 `recovered`。CSRT 自身报告失败、候选框无效或尺寸异常时不自动重试，以减少重新锁到背景的风险。超过时限进入 `LOST`，停止预测展示和自动重试。
6. 按 R 人工重新定位后新建轨迹段；不使用旧预测强行连接。超过短时重试窗口后需要人工定位，无窗口模式持续记录缺失。

CSRT 更新可能改变其内部模板，外部门限无法撤回这个动作，所以拒绝后不沿用该实例；重试只使用最后接受的画面。CSRT 基线没有独立的外观验证器；运动门限不能发现所有“跟着背景移动”的错误。例如模型被人拿起后，CSRT 可能跟到原位置的背景上，仍需人工检查。两种模式都不包含全局重检测器。

## 可调整参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--appearance` | auto | 从初始框判断是否启用局部亮度对比定位；off 强制使用 CSRT |
| `--gate` | 5.991 | 二维平方马氏距离上限；不是像素距离，也不是目标正确概率 |
| `--measurement-std` | 自动 | 图像中心测量标准差，单位 px；1280 长边及以下为 4，4K 为 12 |
| `--acceleration-std` | 自动 | 每个帧间隔内的随机加速度标准差，单位 px/s²；1280 长边及以下为 800，4K 为 2400 |
| `--initial-velocity-std` | 自动 | 初始速度不确定性，单位 px/s；1280 长边及以下为 200，4K 为 600 |
| `--coast-seconds` | 0.5 | 丢失后最多保留预测和短时重试的时间 |
| `--max-frames` | 不限 | 从初始化帧起最多处理多少帧 |
| `--no-video` | 关闭 | 仅导出数据，用于无视频编码器的环境 |
| `--fallback-fps` | 无 | 视频未提供有效 FPS 时需要显式指定 |

默认噪声标准差按 `max(1, 视频长边 / 1280)` 缩放，避免高分辨率下正常位移被过早拒绝；显式传入的数值按原图单位使用，不再缩放。实际生效值写入 `summary.json`。这是分辨率相关的工程起点，不是概率校准。二维 95% 卡方门限只在线性高斯误差、协方差合理等假设下具有相应覆盖意义；连续图像追踪误差可能相关。碰撞、严重模糊、很小的目标或相机移动可能导致误拒绝或漂移，需要检查原始画面和标注数据。

## 验证和演示

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\make_demo.py test_artifacts\demo
.\.venv\Scripts\python.exe track_video.py test_artifacts\demo\demo.avi --headless --roi 40 100 32 32 --output outputs\demo
```

测试包括已知马氏距离、非等间隔采样、错误候选不污染滤波状态、丢失和人工重建，以及真实 CSRT 在合成纹理方块视频上的追踪、消失处理和导出视频回读。合成视频仅验证软件流程，不代表仿真模型在真实录像中的精度。交互窗口需要人工操作验收。

新增测试覆盖旋转和方向变化、短暂缺失后的恢复、超时后停止自动搜索、多个相似候选时拒绝定位，以及低对比度初始框回退到 CSRT。

复现 `camera_20260910_144729.mp4` 的用户初始化位置：

```powershell
.\.venv\Scripts\python.exe track_video.py camera_20260910_144729.mp4 --start-frame 230 --roi 1092 1904 80 88
```

修复版复测：第 230～268 帧记录到 33 帧图像位置，另外 6 帧留空并在后续恢复；进入花盆、枝叶区域后证据不足，随后进入丢失状态。该结果经过关键帧目视抽查，不是整段人工标注后的准确率。上一段 `144704` 按原框选位置会回退到 CSRT，被人拿起后跟到背景的问题仍未解决。

第一版不包含自动首次发现、长期遮挡后的全局重检测、爆炸识别、隐藏落点推断、地面坐标转换或实际物理量估计。
