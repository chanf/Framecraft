"""Framecraft 视频读取工具。

基于 OpenCV 的视频解码与帧抽取，可独立于 ComfyUI 使用。
所有帧均返回为 RGB 的 numpy uint8 数组，形状 [H, W, 3]，值域 [0, 255]。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


class VideoError(RuntimeError):
    """视频打开或读取失败时抛出。"""


@dataclass(frozen=True)
class VideoInfo:
    path: str
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def open_video(path: str) -> cv2.VideoCapture:
    """打开视频文件，失败时抛出 VideoError。

    路径可为绝对路径，也可为相对 ComfyUI input/ 目录的相对路径；
    这里不做目录拼接，由调用方在 ComfyUI 节点内统一处理。
    """
    if not path or not os.path.exists(path):
        raise VideoError(f"视频文件不存在: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f"无法打开视频文件: {path}")
    return cap


def get_video_info(path: str) -> VideoInfo:
    """读取视频元信息（不读取任何帧像素）。"""
    cap = open_video(path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps and fps > 0:
            duration = frame_count / fps
        else:
            duration = 0.0
        return VideoInfo(
            path=path,
            filename=os.path.basename(path),
            width=max(width, 0),
            height=max(height, 0),
            fps=float(fps) if fps and fps > 0 else 0.0,
            frame_count=max(frame_count, 0),
            duration_seconds=duration,
        )
    finally:
        cap.release()


def read_frame_rgb(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    """读取指定索引的单帧，返回 RGB numpy 数组 [H, W, 3] uint8；越界或失败返回 None。"""
    if frame_index < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_first_frame(path: str) -> Tuple[np.ndarray, VideoInfo]:
    """读取视频首帧。"""
    cap = open_video(path)
    try:
        info = get_video_info(path)
        if info.frame_count == 0:
            raise VideoError(f"视频没有可读帧: {path}")
        frame = read_frame_rgb(cap, 0)
        if frame is None:
            raise VideoError(f"读取首帧失败: {path}")
        return frame, info
    finally:
        cap.release()


def read_last_frame(path: str) -> Tuple[np.ndarray, VideoInfo]:
    """读取视频末帧。

    优先用帧索引定位（快路径），某些编码定位不准时回退到逐帧读取直到末尾。
    """
    cap = open_video(path)
    try:
        info = get_video_info(path)
        if info.frame_count == 0:
            raise VideoError(f"视频没有可读帧: {path}")

        last_index = info.frame_count - 1
        frame = read_frame_rgb(cap, last_index)
        if frame is not None:
            # 简单校验：定位读取通常可信，直接返回。
            return frame, info

        # 回退：从头开始逐帧读取，保留最后一帧。
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        last_frame: Optional[np.ndarray] = None
        while True:
            ok, raw = cap.read()
            if not ok or raw is None:
                break
            last_frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if last_frame is None:
            raise VideoError(f"读取末帧失败: {path}")
        return last_frame, info
    finally:
        cap.release()


def read_frames(
    path: str,
    frame_step: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    max_frames: int = -1,
) -> Tuple[List[np.ndarray], VideoInfo]:
    """按间隔采样读取一组帧。

    参数:
        frame_step: 采样间隔，每 step 帧取一帧（>=1）。
        start_frame: 起始帧索引（>=0）。
        end_frame: 结束帧索引（含），-1 表示到视频末尾。
        max_frames: 最多输出帧数，-1 表示不限制。

    返回:
        (frames, info)，frames 为 RGB numpy 数组列表，每个 [H, W, 3] uint8。
    """
    if frame_step < 1:
        frame_step = 1
    if start_frame < 0:
        start_frame = 0

    cap = open_video(path)
    try:
        info = get_video_info(path)
        if info.frame_count == 0:
            raise VideoError(f"视频没有可读帧: {path}")

        total = info.frame_count
        if end_frame < 0 or end_frame >= total:
            end_frame = total - 1
        if start_frame >= total:
            raise VideoError(
                f"起始帧 {start_frame} 超出视频总帧数 {total}: {path}"
            )

        frames: List[np.ndarray] = []
        index = start_frame
        while index <= end_frame:
            frame = read_frame_rgb(cap, index)
            if frame is None:
                break
            frames.append(frame)
            if max_frames > 0 and len(frames) >= max_frames:
                break
            index += frame_step

        if not frames:
            raise VideoError(f"未从视频中读取到任何帧: {path}")
        return frames, info
    finally:
        cap.release()


def frames_to_tensor_list(frames: List[np.ndarray]):
    """将 RGB uint8 帧列表转为 ComfyUI 兼容的 float32 张量列表，每个 [1, H, W, 3]，值域 [0, 1]。

    优先使用 torch；若环境无 torch（如本机独立测试），退回到 numpy 数组并附带
    `is_torch` 标志，由调用方决定是否堆叠。
    """
    try:
        import torch  # type: ignore

        tensors = []
        for frame in frames:
            arr = frame.astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).unsqueeze(0))
        return tensors, True
    except ImportError:
        arrays = []
        for frame in frames:
            arr = frame.astype(np.float32) / 255.0
            arrays.append(arr[np.newaxis, ...])  # [1, H, W, 3]
        return arrays, False
