# Framecraft

ComfyUI 自定义节点集，用于从视频中提取首帧、末帧，或按间隔拆解为帧序列（IMAGE Batch）。

## 功能

| 节点 | 作用 |
|------|------|
| **Video First Frame** | 提取视频首帧，输出单帧 IMAGE 及视频元信息 |
| **Video Last Frame** | 提取视频末帧，输出单帧 IMAGE 及视频元信息 |
| **Video Frame Splitter** | 按间隔采样拆解视频为帧序列，输出 IMAGE Batch |

所有图像输出均为 ComfyUI 原生 `IMAGE` 张量：`[B, H, W, C]`，`float32`，值域 `[0, 1]`，RGB 顺序。

## 安装

将 `custom_nodes/FramecraftVideoNodes/` 拷贝到 ComfyUI 的 `custom_nodes/` 目录，然后重启 ComfyUI：

```bash
# 在 ComfyUI 根目录下
cp -r /path/to/Framecraft/custom_nodes/FramecraftVideoNodes custom_nodes/
# 重启 ComfyUI
```

节点会出现在菜单分类 `Framecraft/Video` 下。

### 依赖

- `opencv-python`（视频解码）
- `numpy`
- `torch`（ComfyUI 环境中已内置，用于输出 IMAGE 张量）

```bash
pip install opencv-python numpy
```

## 节点说明

### Video First Frame / Video Last Frame

**输入**

| 参数 | 类型 | 说明 |
|------|------|------|
| `video` | `STRING` | 视频文件路径（绝对路径，或相对 ComfyUI `input/` 目录的路径） |

**输出**

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | `IMAGE` | 单帧图像 `[1, H, W, 3]` |
| `frame_count` | `INT` | 视频总帧数 |
| `fps` | `FLOAT` | 视频帧率 |
| `width` | `INT` | 视频宽度 |
| `height` | `INT` | 视频高度 |
| `filename` | `STRING` | 源视频文件名 |

### Video Frame Splitter

**输入**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `video` | `STRING` | — | 视频文件路径 |
| `frame_step` | `INT` | `1` | 采样间隔，每 `step` 帧取一帧（≥1） |
| `start_frame` | `INT` | `0` | 起始帧索引（≥0） |
| `end_frame` | `INT` | `-1` | 结束帧索引（含），`-1` 表示到视频末尾 |
| `max_frames` | `INT` | `-1` | 最多输出帧数，`-1` 表示不限制 |

**输出**

| 参数 | 类型 | 说明 |
|------|------|------|
| `images` | `IMAGE` | 帧序列 `[N, H, W, 3]`（N 为实际采样帧数） |
| `frame_count` | `INT` | 实际输出帧数 |
| `total_frames` | `INT` | 视频总帧数 |
| `fps` | `FLOAT` | 视频帧率 |
| `width` | `INT` | 视频宽度 |
| `height` | `INT` | 视频高度 |
| `filename` | `STRING` | 源视频文件名 |

## 使用示例

1. **首帧/末帧**：`Load Video`（或手动输入视频路径）→ `Video First Frame` / `Video Last Frame` → `Preview Image` / `Save Image`
2. **拆解关键帧**：视频路径 → `Video Frame Splitter`（`frame_step=30` 每秒取一帧，假设 30fps）→ 输出帧序列 → 接后续图像处理节点（如 `Save Image` 等）

## 本机逻辑验证

包含 `test_video_nodes.py`，可在无 ComfyUI 环境下用 OpenCV 验证帧抽取逻辑：

```bash
python3 custom_nodes/FramecraftVideoNodes/test_video_nodes.py
```

## 文件结构

```
Framecraft/
├── README.md                        # 本文件
├── design.md                        # 设计文档
└── custom_nodes/
    └── FramecraftVideoNodes/
        ├── __init__.py              # ComfyUI 节点注册入口
        ├── video_frame_nodes.py     # 三个节点类实现
        ├── video_utils.py           # OpenCV 视频读取工具
        ├── test_video_nodes.py      # 独立逻辑测试脚本（无需 ComfyUI）
        └── README.md                # 节点目录说明
```

## 注意事项

- 视频路径必须为本机可访问的文件路径；ComfyUI 的 `Load Video` 节点输出的路径可直接接入本节点的 `video` 输入。
- 某些视频编码对“按索引定位”支持不佳，`Video Last Frame` 在定位失败时会自动回退到逐帧读取直到末尾，保证能取到最后一帧。
- 输出图像为 RGB；如需 BGR，可在后续节点中自行转换。

## 许可

MIT