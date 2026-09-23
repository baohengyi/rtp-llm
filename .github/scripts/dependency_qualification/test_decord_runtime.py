"""Linux dependency qualification; this is not RTP-LLM or GPU acceptance."""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import decord
import numpy as np
import pytest


@pytest.fixture(scope="session")
def media():
    root = Path(os.environ["DECORD_QUALIFICATION_ROOT"]).resolve()
    assert platform.system() == "Linux"
    assert platform.machine() == "x86_64"
    assert sys.prefix != sys.base_prefix
    assert Path(decord.__file__).resolve().is_relative_to(Path(sys.prefix))
    assert Path(np.__file__).resolve().is_relative_to(Path(sys.prefix))
    h264 = root / "h264.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc=size=96x64:rate=10:duration=2", "-c:v", "libx264", "-threads", "1",
        "-pix_fmt", "yuv420p", str(h264),
    ], check=True)
    # A lossless RGB fixture gives a known pixel oracle independent of decord.
    pixels = np.arange(20 * 64 * 96 * 3, dtype=np.uint32).reshape(20, 64, 96, 3)
    pixels = ((pixels * 17 + 31 + np.arange(20, dtype=np.uint32)[:, None, None, None] * 13) % 256).astype(np.uint8)
    raw = root / "expected.rgb"
    raw.write_bytes(pixels.tobytes())
    lossless = root / "lossless.mkv"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", "96x64", "-framerate", "10",
        "-i", str(raw), "-c:v", "ffv1", "-threads", "1", "-pix_fmt", "bgr0",
        str(lossless),
    ], check=True)
    decoded = root / "ffmpeg-decoded.rgb"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(lossless),
        "-f", "rawvideo", "-pix_fmt", "rgb24", str(decoded),
    ], check=True)
    assert decoded.read_bytes() == pixels.tobytes()
    return root, h264, lossless, pixels


def test_native_library_bytes_and_origin(media):
    root, *_ = media
    proof = json.loads((root / "candidate-integrity.json").read_text())
    site = Path(decord.__file__).resolve().parent.parent
    assert len(proof["native_payload"]) == 18
    for item in proof["native_payload"]:
        path = site / item["path"]
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    from decord._ffi.base import _LIB
    assert Path(_LIB._name).resolve() == site / "decord/libdecord.so"


def test_h264_frame_count_shape_and_timestamps(media):
    _, h264, *_ = media
    video = decord.VideoReader(str(h264), ctx=decord.cpu(0), num_threads=1)
    assert len(video) == 20
    frames = video.get_batch(list(range(20))).asnumpy()
    assert frames.shape == (20, 64, 96, 3)
    assert frames.dtype == np.uint8
    assert video.get_avg_fps() == 10.0
    timestamps = video.get_frame_timestamp(list(range(20)))
    assert timestamps.shape == (20, 2)
    assert np.all(timestamps[:, 1] > timestamps[:, 0])
    assert np.all(np.diff(timestamps[:, 0]) > 0)


def test_h264_sequential_equals_batch(media):
    _, h264, *_ = media
    sequential = decord.VideoReader(str(h264), ctx=decord.cpu(0), num_threads=1)
    expected = np.stack([frame.asnumpy() for frame in sequential])
    batched = decord.VideoReader(str(h264), ctx=decord.cpu(0), num_threads=1)
    np.testing.assert_array_equal(batched.get_batch(list(range(20))).asnumpy(), expected)


def test_h264_seek_and_duplicates(media):
    _, h264, *_ = media
    video = decord.VideoReader(str(h264), ctx=decord.cpu(0), num_threads=1)
    expected = video.get_batch(list(range(20))).asnumpy()
    indices = [19, 0, 11, 11, 2, 18, 1, 0]
    np.testing.assert_array_equal(video.get_batch(indices).asnumpy(), expected[indices])
    for index in (0, 19, 4, 10):
        video.seek_accurate(index)
        np.testing.assert_array_equal(video.next().asnumpy(), expected[index])


def test_lossless_pixels_equal_reference(media):
    _, _, lossless, pixels = media
    video = decord.VideoReader(str(lossless), ctx=decord.cpu(0), num_threads=1)
    assert len(video) == 20
    np.testing.assert_array_equal(video.get_batch(list(range(20))).asnumpy(), pixels)


def test_file_like_matches_path(media):
    _, h264, *_ = media
    path_video = decord.VideoReader(str(h264), ctx=decord.cpu(0), num_threads=1)
    memory_video = decord.VideoReader(io.BytesIO(h264.read_bytes()), ctx=decord.cpu(0), num_threads=1)
    assert len(memory_video) == len(path_video) == 20
    indices = list(range(20))
    np.testing.assert_array_equal(memory_video.get_batch(indices).asnumpy(),
                                  path_video.get_batch(indices).asnumpy())
