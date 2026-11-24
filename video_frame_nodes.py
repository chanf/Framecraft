"""Framecraft ComfyUI 自定义节点：视频帧提取与拆解。

提供三个节点：
- VideoFirstFrame：提取视频首帧
- VideoLastFrame：提取视频末帧
- VideoFrameSplitter：按间隔拆解为帧序列（IMAGE Batch）

输出均为 ComfyUI 原生 IMAGE 张量 [B, H, W, C]，float32，值域 [0, 1]，RGB 顺序。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - 无 ComfyUI 环境时 torch 可能缺失
    torch = None  # type: ignore

from .video_utils import (
    VideoError,
    VideoInfo,
    frames_to_tensor_list,
    get_video_info,
    read_first_frame,
    read_frames,
    read_last_frame,
)

# 当环境无 torch 时，允许节点类仍被导入（以便本机静态检查），但执行会报错。
_HAS_TORCH = torch is not None


def _normalize_video_input(video):
    """把 ComfyUI VIDEO 类型的输入归一化为视频文件路径字符串。

    ComfyUI 的 VIDEO 类型有两种表现：
    1. 字符串路径（绝对路径或相对 ComfyUI input/ 目录的相对路径）；
    2. 视频对象（如 `VideoFromComponents`），只暴露元信息，没有直接路径。
    对于第 2 种情况，必须先由生成该视频的节点将其写出为临时文件，
    否则本节点无法用 OpenCV 读取；这里返回 None 并附带清晰报错信息。
    """
    if video is None:
        return None
    if isinstance(video, str):
        return video
    # 非字符串：可能是 VideoFromComponents 等视频对象，尝试提取路径属性
    for attr in ("filename", "path", "temp", "temp_path", "video_path"):
        if hasattr(video, attr):
            value = getattr(video, attr)
            if isinstance(value, str) and value:
                return value
    raise TypeError(
        "Framecraft: `video` 输入应为字符串路径，当前为 %s。"
        " 请确认上游节点（如 Load Video / Minimax H3 输出）连入的是视频文件路径，"
        " 而非 VIDEO 对象本身。若上游仅输出 VIDEO 对象，需先由该节点将视频写出为文件。"
        % type(video).__name__
    )


def _to_comfy_image(frames: List[np.ndarray]):
    """把 RGB uint8 帧列表转成 ComfyUI IMAGE 张量 [N, H, W, 3] float32 [0,1]。"""
    tensors, _ = frames_to_tensor_list(frames)
    if torch is None:
        raise RuntimeError("未安装 torch，无法在 ComfyUI 中输出 IMAGE 张量")
    return torch.cat(tensors, dim=0)


class VideoFirstFrame:
    """提取视频首帧。"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "video": ("VIDEO", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "frame_count", "fps", "width", "height", "filename")
    FUNCTION = "extract_first_frame"
    CATEGORY = "Framecraft/Video"

    def extract_first_frame(self, video: str):
        path = _normalize_video_input(video)
        frame, info = read_first_frame(path)
        image = _to_comfy_image([frame])
        return (
            image,
            info.frame_count,
            info.fps,
            info.width,
            info.height,
            info.filename,
        )


class VideoLastFrame:
    """提取视频末帧。"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "video": ("VIDEO", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "frame_count", "fps", "width", "height", "filename")
    FUNCTION = "extract_last_frame"
    CATEGORY = "Framecraft/Video"

    def extract_last_frame(self, video: str):
        path = _normalize_video_input(video)
        frame, info = read_last_frame(path)
        image = _to_comfy_image([frame])
        return (
            image,
            info.frame_count,
            info.fps,
            info.width,
            info.height,
            info.filename,
        )


class VideoFrameSplitter:
    """按间隔拆解视频为帧序列，输出 IMAGE Batch。"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "video": ("VIDEO", {"default": ""}),
                "frame_step": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 1000000, "step": 1}),
                "max_frames": ("INT", {"default": -1, "min": -1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "images",
        "frame_count",
        "total_frames",
        "fps",
        "width",
        "height",
        "filename",
    )
    FUNCTION = "split_frames"
    CATEGORY = "Framecraft/Video"

    def split_frames(
        self,
        video: str,
        frame_step: int = 1,
        start_frame: int = 0,
        end_frame: int = -1,
        max_frames: int = -1,
    ):
        path = _normalize_video_input(video)
        frames, info = read_frames(
            path,
            frame_step=frame_step,
            start_frame=start_frame,
            end_frame=end_frame,
            max_frames=max_frames,
        )
        images = _to_comfy_image(frames)
        return (
            images,
            len(frames),
            info.frame_count,
            info.fps,
            info.width,
            info.height,
            info.filename,
        )


NODE_CLASS_MAPPINGS: Dict[str, object] = {
    "VideoFirstFrame": VideoFirstFrame,
    "VideoLastFrame": VideoLastFrame,
    "VideoFrameSplitter": VideoFrameSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "VideoFirstFrame": "Video First Frame",
    "VideoLastFrame": "Video Last Frame",
    "VideoFrameSplitter": "Video Frame Splitter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
