import pickle
from unittest import TestCase, main

from rtp_llm.config.test.kv_cache_event_test_values import KV_CACHE_EVENT_FIELD_VALUES
from rtp_llm.ops import KVCacheConfig

EVENT_PICKLE_FIELDS = tuple(KV_CACHE_EVENT_FIELD_VALUES)


DISK_CACHE_FIELDS = (
    "enable_memory_cache_disk",
    "memory_cache_disk_paths",
    "memory_cache_disk_size_mb",
    "memory_cache_disk_buffered_io",
    "memory_cache_disk_sync_timeout_ms",
    "enable_gpu_prefix_tree",
    "enable_prefix_tree_memory_cache",
    "enable_legacy_memory_connector_fallback",
    "prefix_tree_memory_state_swa_pool_ratio",
    "enable_independent_group_eviction",
    "load_cache_retry_times",
)
EVENT_FIELD_OFFSET = 57


class KVCacheConfigPickleTest(TestCase):
    def test_event_fields_round_trip_in_current_state(self):
        config = KVCacheConfig()
        config.block_tree_transfer_worker_count = 7
        config.enable_disk_cache = True
        config.disk_cache_paths = "/tmp/cache"
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(config, name, value)
        state = config.__getstate__()
        self.assertEqual(state[:2], ("KVCacheConfig", 7))
        self.assertEqual(state[-5:], tuple(KV_CACHE_EVENT_FIELD_VALUES.values()))
        # setstate must normalize its private slice, not mutate the supplied
        # version-7 tuple retained by another Python reference.
        original_state = tuple(list(state))
        direct = KVCacheConfig.__new__(KVCacheConfig)
        direct.__setstate__(state)
        self.assertEqual(state, original_state)
        self.assertEqual(direct.__getstate__(), original_state)
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.block_tree_transfer_worker_count, 7)
        self.assertTrue(restored.enable_disk_cache)
        self.assertEqual(restored.disk_cache_paths, "/tmp/cache")
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            self.assertEqual(getattr(restored, name), value, name)

    def test_event_pickle_block_follows_declaration_order(self):
        config = KVCacheConfig()
        for name, value in KV_CACHE_EVENT_FIELD_VALUES.items():
            setattr(config, name, value)
        state = config.__getstate__()
        self.assertEqual(len(state), 69)
        self.assertEqual(
            state[64:], tuple(KV_CACHE_EVENT_FIELD_VALUES[name] for name in EVENT_PICKLE_FIELDS)
        )

    def test_unknown_state_sizes_are_rejected(self):
        state = KVCacheConfig().__getstate__()
        for size in (42, 44, 53, 55, 56, 58, 61, 68):
            with self.subTest(size=size), self.assertRaisesRegex(
                RuntimeError, "invalid KVCacheConfig state"
            ):
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(state[:size])

    def test_previous_block_tree_state_uses_event_defaults(self):
        source = KVCacheConfig()
        source.dsv4_fixed_pool_blocks = 17
        source.block_tree_transfer_worker_count = 7
        state = source.__getstate__()
        previous = (state[0], 6, *state[2:-5])
        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(previous)
        self.assertEqual(restored.dsv4_fixed_pool_blocks, 17)
        self.assertEqual(restored.block_tree_transfer_worker_count, 7)
        defaults = KVCacheConfig()
        for name in EVENT_PICKLE_FIELDS:
            self.assertEqual(getattr(restored, name), getattr(defaults, name), name)

    def test_malformed_event_states_are_rejected(self):
        state = KVCacheConfig().__getstate__()
        for malformed in (
            state[:-1],
            state + ("extra",),
            (state[0], 99, *state[2:]),
            ("OtherConfig", *state[1:]),
        ):
            with self.subTest(state=malformed), self.assertRaisesRegex(
                RuntimeError, "invalid KVCacheConfig state"
            ):
                restored = KVCacheConfig.__new__(KVCacheConfig)
                restored.__setstate__(malformed)

    def test_legacy_54_element_state_uses_event_defaults(self):
        source = KVCacheConfig()
        source.enable_memory_cache_disk = True
        source.memory_cache_disk_paths = "/tmp/cache"
        source.load_cache_retry_times = 7
        legacy_state = source.__getstate__()[:54]
        self.assertEqual(54, len(legacy_state))

        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(legacy_state)

        self.assertTrue(restored.enable_memory_cache_disk)
        self.assertEqual("/tmp/cache", restored.memory_cache_disk_paths)
        self.assertEqual(7, restored.load_cache_retry_times)
        defaults = KVCacheConfig()
        for name in KV_CACHE_EVENT_FIELD_VALUES:
            self.assertEqual(getattr(defaults, name), getattr(restored, name), name)

    def test_legacy_43_element_state_uses_disk_and_event_defaults(self):
        source = KVCacheConfig()
        source.enable_memory_cache_disk = True
        source.memory_cache_disk_paths = "/tmp/cache"
        source.load_cache_retry_times = 7
        source.kv_cache_event_publisher_type = "kvcm"
        legacy_state = source.__getstate__()[:43]
        self.assertEqual(43, len(legacy_state))

        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(legacy_state)

        defaults = KVCacheConfig()
        for name in (*DISK_CACHE_FIELDS, *KV_CACHE_EVENT_FIELD_VALUES):
            self.assertEqual(getattr(defaults, name), getattr(restored, name), name)

    def test_legacy_57_element_state_uses_event_defaults(self):
        source = KVCacheConfig()
        source.dsv4_fixed_pool_blocks = 17
        source.dsv4_hca_state_pool_blocks = 19
        source.dsv4_fixed_pool_use_memory = True
        legacy_state = source.__getstate__()[:EVENT_FIELD_OFFSET]

        restored = KVCacheConfig.__new__(KVCacheConfig)
        restored.__setstate__(legacy_state)

        self.assertEqual(17, restored.dsv4_fixed_pool_blocks)
        self.assertEqual(19, restored.dsv4_hca_state_pool_blocks)
        self.assertTrue(restored.dsv4_fixed_pool_use_memory)
        defaults = KVCacheConfig()
        for name in KV_CACHE_EVENT_FIELD_VALUES:
            self.assertEqual(getattr(defaults, name), getattr(restored, name), name)



if __name__ == "__main__":
    main()
