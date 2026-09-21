"""Keep historical C++ test selectors attached to their migrated assertions."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

ROUTES = [
    (
        '//rtp_llm/cpp/cache/connector/remote_connector/test:group_policy_test',
        '//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:group_policy_compatibility_test',
        {
            'GroupPolicyTest.test_init_FullLinearLayerGroupPolicy_success_single_tp',
            'GroupPolicyTest.test_init_FullLinearLayerGroupPolicy_success_two_tp',
            'GroupPolicyTest.test_init_FullLinearLayerGroupPolicy_success_two_full_groups',
            'GroupPolicyTest.test_init_DefaultLayerGroupPolicy_fail_for_duplicate_group',
            'GroupPolicyTest.test_init_FullLayerGroupPolicy_fail_for_empty_full_group',
            'GroupPolicyTest.test_init_FullLayerGroupPolicy_success_for_multiple_full_groups',
            'GroupPolicyTest.test_init_FullLayerGroupPolicy_fail_for_not_empty_other_group',
            'GroupPolicyTest.test_init_FullLinearLayerGroupPolicy_fail_for_not_empty_group',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_one_tp',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_two_tp',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_success_two_tp_two_full_group',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_2',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_2',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_1',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_1',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_one_tp_interval_0',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_success_two_tp_interval_0',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_fail',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedLoadLocations_fail_two_full_group',
            'GroupPolicyTest.test_FullLinearLayerGroupPolicy_filterNeedWriteGroups_fail',
            'GroupPolicyTest.test_FullLayerGroupPolicy_filterNeedLoadLocations_success',
            'GroupPolicyTest.test_FullLayerGroupPolicy_filterNeedWriteGroups_success',
            'GroupPolicyTest.test_DefaultLayerGroupPolicy_filterNeedWriteGroups_success',
        },
    ),
    (
        '//rtp_llm/cpp/cache/connector/remote_connector/test:client_wrapper_test',
        '//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:client_wrapper_compatibility_test',
        {
            'ClientWrapperTest.test_no_need_reinit',
            'ClientWrapperTest.test_no_invalid_addresses',
            'ClientWrapperTest.test_reinit_with_new_addresses',
            'ClientWrapperTest.test_new_address_create_client_first_fail_second_success',
            'ClientWrapperTest.test_registration',
        },
    ),
    (
        '//rtp_llm/cpp/cache/connector/remote_connector/test:remote_connector_internal_test',
        '//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:legacy_remote_connector_internal_test',
        {
            'RemoteConnectorInternalTest.test_genClientConfig',
            'RemoteConnectorInternalTest.test_genLocationSpecInfoMapAndGroups',
            'RemoteConnectorInternalTest.PublishesTagLocalHeterogeneousGroupBlockSizes',
            'RemoteConnectorInternalTest.test_genLocationSpecGroupsScalesLinearly',
            'RemoteConnectorTagIdentityTest.GroupNamesDoNotDependOnNumericGroupOrder',
            'RemoteConnectorTagIdentityTest.FullOnlyPolicyRoutesSameLayerGroupsByTagWithoutHotPathLayoutLookup',
            'RemoteConnectorBlockBufferValidationTest.RejectsAllocatorBufferSizeThatDoesNotMatchTopology',
            'RemoteConnectorTopologyInvariantTest.InitializationRejectsMissingTopology',
        },
    ),
    (
        '//rtp_llm/cpp/cache/connector/memory/test:disk_block_pool_test',
        '//rtp_llm/cpp/cache/block_tree_cache/block_pool/test:disk_block_pool_test',
        {
            'DiskBlockPoolTest.InitPreallocatesFileAndCleansStaleFiles',
            'DiskBlockPoolTest.InitFailsWhenMountPathDoesNotExist',
            'DiskBlockPoolTest.MountGuardAllowsTwoPoolsOnSameMountWithoutDeletingFirst',
            'DiskBlockPoolTest.ReserveCommitAbortAndFreeSlots',
            'DiskBlockPoolTest.RequestRefPreventsReuseUntilReleased',
            'DiskBlockPoolTest.ReadWriteFullSlot',
            'DiskBlockPoolTest.FullPoolReturnsNullopt',
        },
    ),
]


def rule_for(label):
    package, name = label.removeprefix("//").split(":", 1)
    build = ROOT / package / "BUILD"
    rules = [
        node.value
        for node in ast.parse(build.read_text()).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and any(
            kw.arg == "name" and isinstance(kw.value, ast.Constant) and kw.value.value == name
            for kw in node.value.keywords
        )
    ]
    assert len(rules) == 1, label
    rule = rules[0]
    return package, rule.func.id, {kw.arg: kw.value for kw in rule.keywords}


@pytest.mark.parametrize("old_label,new_label,required_cases", ROUTES)
def test_legacy_selector_reaches_all_migrated_cases(old_label, new_label, required_cases):
    old_package, kind, attrs = rule_for(old_label)
    # A test_suite expands to the actual runnable test. An alias is not a
    # substitute: Bazel test does not execute a test selected only via alias.
    assert kind == "test_suite", old_label
    assert ast.literal_eval(attrs["tests"]) == [new_label]
    assert "tags" not in attrs and "target_compatible_with" not in attrs
    new_package, new_kind, new_attrs = rule_for(new_label)
    assert new_kind in {"cc_test", "cc_test_wrapper"}
    assert "//" + old_package + ":__pkg__" in ast.literal_eval(new_attrs["visibility"])
    if "/remote_connector/" in old_label:
        compatibility = new_attrs["target_compatible_with"]
        assert isinstance(compatibility, ast.Call)
        # The old entries excluded only CUDA13. Moving them must not add a
        # new use_remote_kv_cache gate that silently removes other platforms.
        assert ast.literal_eval(compatibility.args[0]) == {
            "//:using_cuda13_arm": ["@platforms//:incompatible"],
            "//:using_cuda13_x86": ["@platforms//:incompatible"],
            "//conditions:default": [],
        }
    cases = set()
    for source in ast.literal_eval(new_attrs["srcs"]):
        path = ROOT / new_package / source
        assert path.is_file(), path
        if path.suffix == ".cc":
            cases.update(
                suite + "." + name
                for suite, name in re.findall(
                    r"^TEST(?:_F|_P)?\(\s*(\w+)\s*,\s*(\w+)\s*\)",
                    path.read_text(), re.MULTILINE,
                )
            )
    assert required_cases <= cases, sorted(required_cases - cases)
