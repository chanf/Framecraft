#!/usr/bin/env python3
"""Framecraft 节点逻辑独立测试脚本。

不依赖 ComfyUI，也不依赖 torch。使用 OpenCV 生成一段彩色测试视频，
再分别验证：
  1. 首帧提取（首帧应为第 0 帧的颜色）
  2. 末帧提取（末帧应为最后一帧的颜色）
  3. 拆解帧（按 frame_step 采样，帧数与颜色应符合预期）

用法：
    cd /Users/feng/Work/Framecraft
    python3 custom_nodes/FramecraftVideoNodes/test_video_nodes.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# 确保能导入同目录下的 video_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_utils import (  # noqa: E402
    VideoError,
    frames_to_tensor_list,
    read_first_frame,
    read_frames,
    read_last_frame,
)

try:
    from video_frame_nodes import _normalize_video_input
except ImportError:
    # 无法导入（无 ComfyUI 环境时 torch 缺失），跳过归一化测试
    _normalize_video_input = None

TEST_VIDEO = "/tmp/framecraft_test_video.mp4"


def make_test_video(path: str, frame_count: int = 12, fps: float = 24.0,
                    width: int = 64, height: int = 64) -> list:
    """生成一段每帧颜色不同的测试视频，返回每帧的 RGB 颜色（供断言用）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    colors = []
    for i in range(frame_count):
        # 每帧用不同的纯色，便于断言
        color = (
            int((i * 37) % 256),
            int((i * 91) % 256),
            int((i * 151) % 256),
        )
        colors.append(color)
        # OpenCV 写入的是 BGR 帧
        frame = np.full((height, width, 3), (color[2], color[1], color[0]), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return colors


def assert_frame_color(frame: np.ndarray, expected_rgb: tuple, label: str):
    """断言帧为近似纯色（允许轻微编码误差，mp4 是有损的）。"""
    mean = frame.reshape(-1, 3).mean(axis=0)
    for actual, expect in zip(mean, expected_rgb):
        assert abs(int(actual) - expect) < 20, (
            f"{label}: 期望颜色分量约 {expect}，实际均值 {actual:.1f}"
        )


def main():
    colors = make_test_video(TEST_VIDEO, frame_count=12, fps=24.0, width=64, height=64)

    # ---------- 1. 首帧 ----------
    frame, info = read_first_frame(TEST_VIDEO)
    assert info.frame_count == 12, f"总帧数应为 12，实际 {info.frame_count}"
    assert abs(info.fps - 24.0) < 1.0, f"fps 应约 24，实际 {info.fps}"
    assert info.width == 64 and info.height == 64
    assert info.filename == "framecraft_test_video.mp4"
    assert_frame_color(frame, colors[0], "首帧")
    print(f"[OK] 首帧：shape={frame.shape}, fps={info.fps}, "
          f"count={info.frame_count}, file={info.filename}")

    # ---------- 2. 末帧 ----------
    frame, info = read_last_frame(TEST_VIDEO)
    assert_frame_color(frame, colors[-1], "末帧")
    assert info.frame_count == 12
    print(f"[OK] 末帧：shape={frame.shape}, 最后一帧颜色正确")

    # ---------- 3. 拆解帧：全量 step=1 ----------
    frames, info = read_frames(TEST_VIDEO, frame_step=1, start_frame=0, end_frame=-1, max_frames=-1)
    assert len(frames) == 12, f"全量拆解应得 12 帧，实际 {len(frames)}"
    for i, f in enumerate(frames):
        assert_frame_color(f, colors[i], f"第 {i} 帧")
    print(f"[OK] 拆解帧 step=1：{len(frames)} 帧，每帧颜色均正确")

    # ---------- 4. 拆解帧：step=3 ----------
    frames, info = read_frames(TEST_VIDEO, frame_step=3, start_frame=0, end_frame=-1, max_frames=-1)
    expected_indices = list(range(0, 12, 3))  # 0, 3, 6, 9
    assert len(frames) == len(expected_indices), (
        f"step=3 应得 {len(expected_indices)} 帧，实际 {len(frames)}"
    )
    for f, idx in zip(frames, expected_indices):
        assert_frame_color(f, colors[idx], f"step=3 第 {idx} 帧")
    print(f"[OK] 拆解帧 step=3：{len(frames)} 帧，索引 {expected_indices}，颜色正确")

    # ---------- 5. 拆解帧：start_frame + end_frame + max_frames ----------
    frames, info = read_frames(
        TEST_VIDEO, frame_step=1, start_frame=2, end_frame=7, max_frames=3
    )
    # 索引 2,3,4 即 3 帧（max_frames 截断）
    assert len(frames) == 3, f"截断应得 3 帧，实际 {len(frames)}"
    for f, idx in zip(frames, [2, 3, 4]):
        assert_frame_color(f, colors[idx], f"截断第 {idx} 帧")
    print(f"[OK] 拆解帧 start=2/end=7/max=3：{len(frames)} 帧，索引 2,3,4，颜色正确")

    # ---------- 6. frames_to_tensor_list 输出形状与值域 ----------
    tensors, is_torch = frames_to_tensor_list(frames[:2])
    assert len(tensors) == 2
    for t in tensors:
        assert t.shape == (1, 64, 64, 3), f"张量形状应为 (1,64,64,3)，实际 {t.shape}"
        assert t.min() >= 0.0 and t.max() <= 1.0, "张量值域应在 [0,1] 之间"
        # torch 环境下是 torch.float32，无 torch 环境下是 np.float32
        if is_torch:
            assert t.dtype == torch.float32, f"张量 dtype 应为 float32，实际 {t.dtype}"
        else:
            assert t.dtype == np.float32, f"张量 dtype 应为 float32，实际 {t.dtype}"
    print("[OK] frames_to_tensor_list：形状 (1,64,64,3)，值域 [0,1]，dtype float32")

    # ---------- 7. 错误情况：不存在的文件 ----------
    try:
        read_first_frame("/tmp/nonexistent_video.mp4")
        assert False, "应抛出 VideoError"
    except VideoError:
        print("[OK] 不存在的视频文件抛出 VideoError")

    # ---------- 8. 错误情况：起始帧超出总帧数 ----------
    try:
        read_frames(TEST_VIDEO, start_frame=100)
        assert False, "应抛出 VideoError"
    except VideoError:
        print("[OK] 起始帧超出总帧数抛出 VideoError")

    # ---------- 9. _normalize_video_input 归一化测试 ----------
    if _normalize_video_input is not None:
        # 字符串路径直接返回
        assert _normalize_video_input("/path/to/video.mp4") == "/path/to/video.mp4"
        assert _normalize_video_input("relative/path.mp4") == "relative/path.mp4"

        # 有路径属性的对象返回该属性
        class FakeVideoWithPath:
            def __init__(self):
                self.filename = "/tmp/fake.mp4"
        assert _normalize_video_input(FakeVideoWithPath()) == "/tmp/fake.mp4"

        # VideoFromComponents 对象抛出 TypeError
        class VideoFromComponents:
            def __init__(self):
                self._components = {}
                self._video_info = {"fps": 24.0}
        try:
            _normalize_video_input(VideoFromComponents())
            assert False, "VideoFromComponents 应抛出 TypeError"
        except TypeError:
            print("[OK] VideoFromComponents 抛出 TypeError")

        # None 抛出 TypeError
        try:
            _normalize_video_input(None)
            assert False, "None 应抛出 TypeError"
        except TypeError:
            print("[OK] None 抛出 TypeError")

    # 清理
    try:
        os.remove(TEST_VIDEO)
    except OSError:
        pass

    print("\n全部测试通过 ✔")


if __name__ == "__main__":
    main()
