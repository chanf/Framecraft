# Framecraft — ComfyUI 视频帧处理自定义节点设计文档

## 1. 概述

为 ComfyUI 开发一组自定义节点，支持对输入视频做三件事：

1. **提取首帧** (`VideoFirstFrame`)
2. **提取末帧** (`VideoLastFrame`)
3. **拆解帧序列** (`VideoFrameSplitter`) — 按间隔采样，输出图像 Batch

三个节点共用一套底层视频读取工具（基于 OpenCV），输出均为 ComfyUI 原生 `IMAGE` 张量（`[B, H, W, C]`，`float32`，`[0,1]`，RGB 顺序），并随同输出视频元信息（帧数、帧率、宽、高、文件名）。

## 2. 环境前提（本机现状）

- 工作目录：`/Users/feng/Work/Framecraft`
- 本机 **无完整 ComfyUI**（`/Users/feng/Work/ComfyUI` 仅含下载脚本与 models，无源码、无 `custom_nodes`）。
- Python 3.9.6，`opencv-python` 4.10.0 可用；`torch` / `decord` 不可用。
- 因此实现阶段优先用 **OpenCV** 做视频解码，并提供独立测试脚本在无 ComfyUI 环境下验证帧抽取逻辑。

## 3. 节点设计

### 3.1 共享输入：视频文件路径或 VIDEO 对象

所有节点共用一个 `video` 输入，类型为 ComfyUI 原生 `VIDEO`：

| 参数 | 类型 | 说明 |
|------|------|------|
| `video` | `VIDEO` | 视频文件绝对路径（`str`，或相对 ComfyUI `input/` 的路径），或 ComfyUI 原生 `VIDEO` 对象（如 `VideoFromComponents`）。连入 `VIDEO` 对象时，节点直接读取 `get_components().images` 张量，无需落地临时文件。 |

### 3.2 节点 A：VideoFirstFrame（首帧）

| 字段 | 值 |
|------|-----|
| 类名 | `VideoFirstFrame` |
| 显示名 | `Video First Frame` |
| 分类 | `Framecraft/Video` |

**输入**：`video`（STRING）

**输出**：
- `image`：IMAGE，形状 `[1, H, W, 3]`
- `frame_count`：INT，视频总帧数
- `fps`：FLOAT，视频帧率
- `width`：INT，视频宽度
- `height`：INT，视频高度
- `filename`：STRING，源视频文件名

**逻辑**：打开视频，读取第 0 帧，转 RGB → 归一化到 `[0,1]` → ComfyUI IMAGE。

### 3.3 节点 B：VideoLastFrame（末帧）

| 字段 | 值 |
|------|-----|
| 类名 | `VideoLastFrame` |
| 显示名 | `Video Last Frame` |
| 分类 | `Framecraft/Video` |

**输入**：`video`（STRING）

**输出**：同节点 A。

**逻辑**：
1. 尝试用 `cv2.CAP_PROP_POS_FRAMES` 定位到 `count - 1` 读取（快路径）。
2. 某些编码定位不准时，回退为逐帧读取直到末尾，保留最后一帧。

### 3.4 节点 C：VideoFrameSplitter（拆解帧）

| 字段 | 值 |
|------|-----|
| 类名 | `VideoFrameSplitter` |
| 显示名 | `Video Frame Splitter` |
| 分类 | `Framecraft/Video` |

**输入**：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `video` | STRING | — | 视频文件路径 |
| `frame_step` | INT | `1` | 采样间隔，每 `step` 帧取一帧（≥1） |
| `start_frame` | INT | `0` | 起始帧索引（≥0） |
| `end_frame` | INT | `-1` | 结束帧索引（-1 表示到视频末尾） |
| `max_frames` | INT | `-1` | 最多输出帧数（-1 表示不限制） |

**输出**：
- `images`：IMAGE，形状 `[N, H, W, 3]`（N 为实际采样帧数）
- `frame_count`：INT，实际输出帧数
- `total_frames`：INT，视频总帧数
- `fps`：FLOAT
- `width`：INT
- `height`：INT
- `filename`：STRING

**逻辑**：从 `start_frame` 开始，每隔 `frame_step` 帧读取一帧，直到 `end_frame` 或视频末尾或达到 `max_frames`；将所有帧堆叠为一个 Batch 张量输出。

## 4. 图像转换规范

OpenCV 读取为 BGR `uint8` `[0,255]`，转换为 ComfyUI IMAGE：

```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0  # [H, W, 3], [0,1]
```

Batch 堆叠：`torch.stack(frames, dim=0)` → `[N, H, W, 3]`。

## 5. 文件结构

```
Framecraft/
├── README.md                        # 项目说明与安装
├── docs/
│   └── design.md                    # 本文件
├── __init__.py                      # ComfyUI 节点注册入口
├── video_frame_nodes.py             # 三个节点类实现
├── video_utils.py                   # OpenCV 视频读取工具（可独立于 ComfyUI 使用）
├── test_video_nodes.py              # 独立逻辑测试脚本（无需 ComfyUI，本机可直接运行）
└── requirements.txt                 # opencv-python, numpy, torch（可选）
```

## 6. 注册入口（__init__.py）

```python
from .video_frame_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

## 7. 验证计划

因为本机无 ComfyUI，验证分两阶段：

1. **本机（当前环境）**：运行 `test_video_nodes.py`，用 OpenCV 对一段测试视频分别验证首帧、末帧、拆解帧的输出形状、数值范围、帧数正确性；如无测试视频，用 OpenCV 生一段彩色测试视频再验证。
2. **ComfyUI 环境就绪后**：将根目录下的插件文件（`__init__.py`、`video_frame_nodes.py`、`video_utils.py`、`requirements.txt`）拷入 `custom_nodes/FramecraftVideoNodes/`，重启 ComfyUI，在工作流中连接 `Load Video` → 节点 → `Preview Image` 确认图像正常显示。

## 8. 边界情况与错误处理

- 视频路径不存在或不可读 → 抛出清晰的异常，ComfyUI 节点显示红色错误。
- 视频帧数为 0 或损坏 → 抛出异常。
- `start_frame` 超出总帧数 → 抛出“起始帧超过视频长度”。
- `frame_step < 1` → 强制为 1。
- `end_frame` 为 -1 时视为视频末尾。

---

下一步：按此设计实现 `video_utils.py`、`video_frame_nodes.py`、`__init__.py`，并编写 `test_video_nodes.py` 做本机逻辑验证。