import importlib.util
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_device_resource_module():
    path = PROJECT_ROOT / "rtp_llm" / "test" / "utils" / "device_resource.py"
    spec = importlib.util.spec_from_file_location("_device_resource_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


device_resource = _load_device_resource_module()


class DeviceResourceMainContractTest(TestCase):
    def test_default_requires_one_gpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 1)

        resource = object.__new__(device_resource.DeviceResource)
        with (
            patch.dict(
                os.environ,
                {"RTP_REMOTE_SESSION_ID": "remote-test"},
                clear=True,
            ),
            patch.object(device_resource.subprocess, "run") as run,
        ):
            run.return_value.returncode = 0
            run.return_value.stdout = "151859\n"
            self.assertTrue(resource._has_excess_preexisting_memory("0"))
            run.return_value.stdout = "512\n"
            self.assertFalse(resource._has_excess_preexisting_memory("0"))

        resource._candidate_gpu_groups = lambda: [[0]]
        resource._gpu_bad_until = {}
        resource.gpu_status_root_path = "/tmp/test-gpu-status"
        resource.required_gpu_count = 1
        resource._has_zombie_gpu_contexts = lambda gpu_id: False
        resource._has_non_session_live_cuda_pids = lambda gpu_id: False
        resource._has_excess_preexisting_memory = lambda gpu_id: True
        with (
            patch.object(device_resource, "FileLock"),
            self.assertRaisesRegex(
                device_resource.GpuLockTimeoutError, "pre-existing memory"
            ),
        ):
            resource._lock_gpus()

    def test_default_uses_cpu_when_no_device_is_available(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                device_resource._get_required_gpu_count(device_available=False), 0
            )

    def test_explicit_gpu_request_is_preserved_without_a_device(self):
        with patch.dict(os.environ, {"GPU_COUNT": "1"}, clear=True):
            self.assertEqual(
                device_resource._get_required_gpu_count(device_available=False), 1
            )

    def test_gpu_count_zero_is_explicit_no_lock_opt_out(self):
        with patch.dict(os.environ, {"GPU_COUNT": "0"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 0)

    def test_gpu_count_takes_precedence_over_world_size(self):
        with patch.dict(os.environ, {"GPU_COUNT": "2", "WORLD_SIZE": "4"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 2)

    def test_world_size_used_when_gpu_count_absent(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True):
            self.assertEqual(device_resource._get_required_gpu_count(), 4)
