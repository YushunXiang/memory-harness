from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from memory_harness.research_candidates import SCHEMA_VERSION
from memory_harness.research_candidates import validate_source_audited_candidates


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CATALOG = PROJECT_ROOT / "configs" / "source_audited_candidates.json"
LAMEM_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-lamem-vla-paper-release-audit.json"
)
VQ_MEMORY_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-vq-memory-paper-release-audit.json"
)
HYMES_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-hymes-paper-release-audit.json"
RTCF_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-rtcf-paper-release-audit.json"
ATLASVLA_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-atlasvla-paper-release-audit.json"
)
ONEVOMEMORY_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-onevomemory-paper-release-audit.json"
)
STREAMING_GRPO_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-streaming-grpo-memory-source-audit.json"
)
RBVLA_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-rbvla-paper-release-audit.json"
STEMVLA_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-stemvla-paper-release-audit.json"
)
HIME_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-hime-source-audit.json"
HIMEM_WAM_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-himem-wam-source-audit.json"
WORLDSCAPE_POLICY2_AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "2026-08-16-worldscape-policy2-paper-release-audit.json"
)
MEM_WORLD_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-mem-world-paper-release-audit.json"
)
MEMORA_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-memora-paper-release-audit.json"
MIMIR_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-mimir-paper-release-audit.json"
GESTO_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-gesto-paper-release-audit.json"
SMA_AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "2026-08-16-spatial-memory-agent-paper-release-audit.json"
)
R4DSG_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-r4dsg-paper-release-audit.json"
DREAMFLY_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-dreamfly-paper-release-audit.json"
)
STREAMFLOW_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-streamflow-paper-release-audit.json"
)
DRIVEVLA_M0_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-drivevla-m0-source-audit.json"
)
CONSOLIDATOR_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-consolidator-source-audit.json"
)
STREAMTTT_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-streamttt-paper-release-audit.json"
)
QCR_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-qcr-paper-release-audit.json"
ECHOVLA_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-echovla-paper-release-audit.json"
)
DREAM_SPATIAL_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-dream-spatial-source-audit.json"
)
G05_MEM_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-g05-mem-source-audit.json"
BRIDGEVLA_PLUS_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-bridgevla-plus-source-audit.json"
)
SERF_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-serf-source-audit.json"
PHYSMEM_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-physmem-source-audit.json"
VERMEM_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-vermem-source-audit.json"
SIGNALS_TO_STRUCTURE_AUDIT = (
    PROJECT_ROOT
    / "artifacts"
    / "2026-08-16-signals-to-structure-paper-release-audit.json"
)
HIERARCHICAL_MEMORY_THEORY_AUDIT = (
    PROJECT_ROOT / "artifacts" / "2026-08-16-hierarchical-memory-theory-audit.json"
)
ALMA_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-alma-source-audit.json"
MEMEVOLVE_AUDIT = PROJECT_ROOT / "artifacts" / "2026-08-16-memevolve-source-audit.json"


def test_source_audited_candidates_are_distinct_and_non_executable() -> None:
    payload = validate_source_audited_candidates(CATALOG)

    assert payload["schema_version"] == SCHEMA_VERSION
    candidates = {candidate["id"]: candidate for candidate in payload["candidates"]}
    assert set(candidates) == {
        "aha_wam_dual_rate_planner_context",
        "bridgevla_plus_canonical_pointcloud_anchor",
        "chronos_selective_ssm",
        "g05_mem_temporal_spatial_video",
        "gmp_error_calibrated_read_gate",
        "halo_vqa_sparse_attention",
        "kcvla_task_phase_event_keyframes",
        "memoryvla_dual_stream_pcmb",
        "muvla_recurrent_tokens",
        "nativemem_action_supervised_tokens",
        "optimusvla_cross_episode_action_prior",
        "physmem_verified_physical_principles",
        "robomme_perceptual_patch_memory",
        "serf_shared_robot_environment_neural_points",
        "tfp_ltc_action_adaln",
        "trace_trajectory_routed_slots",
        "vlapro_parameter_memory",
    }
    assert {candidate["payload_family"] for candidate in candidates.values()} == {
        "asynchronous_layerwise_planner_kv_context",
        "observation_action_kv",
        "recurrent_state",
        "recurrent_token_state",
        "continuous_time_belief_state",
        "bounded_multiview_raw_frame_window",
        "dual_stream_perceptual_cognitive_memory",
        "action_supervised_native_visual_tokens",
        "trajectory_action_prior",
        "temporal_visual_patch_memory",
        "task_phase_keyframe_history",
        "trajectory_addressed_evidence_slots",
        "parameter_memory",
        "error_calibrated_read_gate",
        "viewpoint_aligned_canonical_pointcloud_anchor",
        "articulated_robot_environment_relational_neural_point_state",
        "verified_physical_principle_hypothesis_lifecycle",
    }
    assert all(
        candidate["implementation_status"] == "not_executable"
        and candidate["executable_architecture_alias"] is None
        for candidate in candidates.values()
    )
    assert all(
        operator["inputs"] and operator["output"]
        for candidate in candidates.values()
        for operator in candidate["operators"]
    )


def test_bridgevla_plus_candidate_keeps_only_distinct_3d_spatial_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "bridgevla_plus_canonical_pointcloud_anchor"
    )

    assert candidate["payload_family"] == (
        "viewpoint_aligned_canonical_pointcloud_anchor"
    )
    assert candidate["entry_gate"] == (
        "fixed_episode_memory_utility_gate_and_calibrated_3d_input_gate"
    )
    assert {
        "calibrated_deployment_visible_rgbd_and_world_frame_contract",
        "matched_full_temporal_only_spatial_only_and_none_training_factorial",
        "memory_specific_weights_and_joint_policy_finetuning",
        "source_checkpoint_reproduction_before_pi05_port",
    } <= set(candidate["requirements"])
    assert {
        "episode_initial_colored_pointcloud_anchor_writer",
        "episode_frozen_canonical_pointcloud_anchor_store",
        "coarse_waypoint_aligned_pointcloud_rerender_retriever",
        "language_conditioned_spatial_anchor_token_encoder",
        "corresponding_view_fine_stage_spatial_cross_attention_utilizer",
        "episode_boundary_canonical_pointcloud_anchor_reset",
        "joint_spatial_anchor_policy_finetuning",
    } == {operator["name"] for operator in candidate["operators"]}
    assert all(
        "temporal" not in operator["name"] and "subgoal" not in operator["name"]
        for operator in candidate["operators"]
    )

    audit = json.loads(BRIDGEVLA_PLUS_AUDIT.read_text(encoding="utf-8"))
    assert audit["official_source"]["commit"] == (
        "8855333c86de4842df40671b6f222d2ffb52e50b"
    )
    assert audit["official_checkpoint"]["put_back_block"]["lfs_sha256"] == (
        "f0ee4d9cb1af253ec66d10eea42d96362265c011df5f34477afde220c2e7264e"
    )
    assert audit["reported_results_percent"]["rmbench"] == {
        "episodes_per_task": 100,
        "full": 96.0,
        "without_spatial": 95.4,
        "without_temporal": 21.3,
        "no_memory": 18.9,
        "put_back_block": {
            "full": 100.0,
            "without_spatial": 100.0,
            "without_temporal": 38.0,
            "no_memory": 1.0,
        },
        "cover_blocks": {
            "full": 99.0,
            "without_spatial": 91.0,
            "without_temporal": 5.0,
            "no_memory": 3.0,
        },
    }
    assert audit["source_smoke"]["spatial"]["fully_masked_reference_is_exact_identity"]
    assert audit["source_smoke"]["temporal"]["finite_memory_token_gradient"]
    assert audit["reproduction_boundary"]["not_a_parameter_free_plugin"]
    assert audit["decision"]["add_to_source_audited_catalog"]


def test_serf_candidate_keeps_robot_environment_relational_axis_composable() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "serf_shared_robot_environment_neural_points"
    )

    assert candidate["payload_family"] == (
        "articulated_robot_environment_relational_neural_point_state"
    )
    assert candidate["entry_gate"] == (
        "fixed_episode_memory_utility_gate_and_mobile_calibrated_3d_robot_state_gate"
    )
    assert {
        "calibrated_deployment_visible_rgbd_camera_and_world_frame_contract",
        "deployable_instance_identity_tracking_or_explicit_privileged_input_label",
        "independently_selectable_robot_base_end_effector_robot_only_environment_only_and_global_branches",
        "matched_image_static_dynamic_environment_and_robot_environment_training_factorial",
        "matched_each_token_group_all_groups_and_none_training_factorial",
        "budget_matched_pi05_joint_training",
    } <= set(candidate["requirements"])
    operator_names = {operator["name"] for operator in candidate["operators"]}
    assert {
        "proprioceptive_forward_kinematics_robot_point_writer",
        "shared_articulated_robot_environment_neural_point_store",
        "robot_base_multiscale_ball_query_retriever",
        "bilateral_end_effector_ball_query_retriever",
        "robot_only_neural_point_retriever",
        "environment_only_neural_point_retriever",
        "global_robot_environment_neural_point_retriever",
        "branch_specific_point_transformer_attention_pool_encoder",
        "pi05_spatial_relational_map_prefix_utilizer",
        "episode_scene_robot_environment_map_reset",
        "map_tokenizer_action_lora_joint_finetuning",
    } <= operator_names

    audit = json.loads(SERF_AUDIT.read_text(encoding="utf-8"))
    assert audit["official_source"]["policy_repository"]["commit"] == (
        "ea27b7aa753cf7da6def975846ccb5d3180e46f7"
    )
    assert audit["official_source"]["mapping_repository"]["commit"] == (
        "e68f39a45a17bfeef2c12342b715ff6aa74f42cb"
    )
    assert audit["official_release"]["policy_checkpoint_revision"] == (
        "1469ac6c84044a2f0613dec461b371dbebce3235"
    )
    assert audit["reported_results_percent"]["matched_deltas"] == {
        "full_minus_pi05_finetuned_mean_pp": 14.7,
        "dynamic_environment_minus_pi05_finetuned_mean_pp": 11.4,
        "full_minus_dynamic_environment_mean_pp": 3.3,
        "dynamic_environment_minus_static_environment_mean_pp": 1.4,
        "full_minus_dynamic_environment_by_task_pp": [5.6, 1.1, 3.1],
    }
    assert audit["source_smoke"]["full_output_shape"] == [1, 8, 16]
    assert audit["source_smoke"]["environment_only_output_shape"] == [1, 6, 16]
    assert audit["source_smoke"]["finite_input_feature_gradients"]
    assert audit["distinctness_and_deduplication"][
        "dynamic_environment_map_not_counted_twice"
    ]
    assert audit["reproduction_boundary"]["not_a_parameter_free_plugin"]
    assert audit["decision"]["add_to_source_audited_catalog"]


def test_physmem_candidate_keeps_verified_principles_distinct_and_fail_closed() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "physmem_verified_physical_principles"
    )

    assert candidate["payload_family"] == (
        "verified_physical_principle_hypothesis_lifecycle"
    )
    assert candidate["entry_gate"] == (
        "fixed_episode_memory_utility_gate_then_action_level_physical_outcome_and_safe_experiment_gate"
    )
    assert {
        "paper_source_parity_for_targeted_verification_and_promotion",
        "bounded_priority_eviction_ttl_and_decay_contract",
        "resume_rebinds_every_component_to_loaded_stores",
        "deployable_action_level_success_failure_and_symbolic_state_contract",
        "safe_action_constraint_interface_for_targeted_experiments",
        "no_oracle_action_or_evaluation_state_in_write_consolidation_or_retrieval",
        "matched_none_direct_retrieval_no_verification_no_resonance_no_forgetting_no_working_memory_ablations",
    } <= set(candidate["requirements"])
    assert {
        "action_attributed_physical_outcome_experience_writer",
        "principle_resonance_prediction_error_learning_gate",
        "typed_physical_hypothesis_generator",
        "support_contradiction_action_attribution_writer",
        "targeted_physical_hypothesis_experiment_controller",
        "verified_hypothesis_to_physical_principle_promoter",
        "evidence_linked_verified_physical_principle_store",
        "symbolic_then_semantic_physical_principle_retriever",
        "tentative_hypothesis_verified_principle_prompt_utilizer",
        "principle_decay_and_evidence_folding_lifecycle",
        "cross_episode_physical_knowledge_session_lifecycle",
    } <= {operator["name"] for operator in candidate["operators"]}

    audit = json.loads(PHYSMEM_AUDIT.read_text(encoding="utf-8"))
    assert audit["official_source"]["commit"] == (
        "bb2b9bbf0c9561e5e1279437cce154419edc8007"
    )
    assert audit["official_source"]["source_tests_present"] is False
    assert audit["source_smoke"]["quickstart"] == {
        "exit_code": 0,
        "episodes": 200,
        "experiences": 200,
        "hypotheses": 8,
        "principles": 3,
        "final_rolling_success_percent": 45,
        "finding": (
            "the released synthetic stochastic grasp demo executes end to end and "
            "can form promoted principles, but it is neither deterministic evidence "
            "of improvement nor a paper benchmark reproduction"
        ),
    }
    assert audit["source_smoke"]["bounded_writer"] == {
        "configured_max": 2,
        "actual_size_after_three_forced_writes": 3,
    }
    assert audit["source_smoke"]["active_verification"] == {
        "conditions_in_first_plan": 1,
        "action_constraint": None,
        "status_after_one_success": "verified",
        "verification_history_entries": 1,
        "ready_for_promotion": False,
        "second_get_returns_same_completed_plan": True,
        "third_get_returns": None,
    }
    assert not any(audit["source_smoke"]["resume_store_identity"].values())
    assert audit["reported_results_percent"]["matched_component_ablation"][
        "without_verification"
    ] == {
        "easy": 85,
        "medium": 64,
        "hard": 27,
        "relative_tokens": 0.85,
    }
    assert audit["distinctness_and_deduplication"]["retained_new_axes"] == [
        "evidence-linked typed physical principle payload",
        "action-level support and contradiction attribution",
        "targeted hypothesis experiment controller",
        "verified promotion and refutation lifecycle",
        "experience folding after principle promotion",
    ]
    assert audit["decision"]["add_to_source_audited_catalog"]
    assert audit["decision"]["executable_registry_status"].startswith("not_added")


def test_vermem_keeps_controller_credit_axes_without_duplicate_payload() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(VERMEM_AUDIT.read_text(encoding="utf-8"))

    assert audit["official_source"]["commit"] == (
        "4782751c79faa08421a27c23b4d02c591bc3357d"
    )
    assert audit["official_source"]["license_file"] is None
    assert audit["official_source"]["source_tests_present"] is False
    assert audit["paper_contract"]["seven_agent_operations"] == [
        "LTM_ADD",
        "LTM_UPDATE",
        "LTM_DELETE_SOFT",
        "STM_RETRIEVE_FROM_LTM",
        "STM_FILTER",
        "STM_SELECT_EPISODE_CONTEXT",
        "STM_SUMMARIZE",
    ]
    assert audit["source_smoke"]["direct_base_reward_import"] == {
        "passed": False,
        "error": "AttributeError: module 'importlib' has no attribute 'util'",
    }
    assert audit["source_smoke"]["verifier_credit_stack_after_import_workaround"] == {
        "passed": False,
        "failures": ["ModuleNotFoundError: No module named 'agent_memory'"],
        "local_verifier_invoked": False,
        "global_verifier_invoked": False,
        "rl_credit_assigner_invoked": False,
    }
    assert audit["source_smoke"]["base_reward_calibration"][
        "failed_ordering_checks"
    ] == [
        "correct_delete_patch_gt_partial_delete_patch",
        "partial_delete_patch_gt_empty_delete_patch",
        "grounded_rationale_gt_generic_same_action",
        "valid_safe_json_gt_markdown_wrapper",
    ]
    assert audit["reported_results"]["qwen2_5_7b"]["vermem_full"][-1] == 48.01
    assert audit["reported_results"]["qwen2_5_7b"]["vermem_no_verify"][-1] == (41.76)
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    assert audit["distinctness_and_deduplication"]["retained_nonduplicate_axes"] == [
        "one policy over typed LTM and STM operations",
        "realized transition local training verifier",
        "terminal memory consistency global training verifier",
        "operation-normalized hierarchical local and global credit",
        "LTM-to-STM-to-joint curriculum",
    ]
    assert audit["decision"]["retain_source_level_operators"] == [
        "unified_ltm_stm_atomic_memory_operation_policy",
        "versioned_update_soft_delete_memory_transaction_lifecycle",
        "realized_memory_transition_local_training_verifier",
        "terminal_memory_consistency_global_training_verifier",
        "operation_normalized_local_global_hierarchical_credit_assignment",
        "ltm_then_stm_then_joint_memory_operation_curriculum",
    ]
    assert audit["decision"]["add_new_payload_family"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["executable_registry_status"].startswith("not_added")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_signals_to_structure_adds_capacity_protocol_not_duplicate_memory() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(SIGNALS_TO_STRUCTURE_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["official_html_sha256"] == (
        "1576e4dfd21d1024f1ab951263a784bf0d5203f92140a632f87731c93baac70d"
    )
    assert audit["paper_contract"]["common_base_memory"] == (
        "all five conditions receive the same rolling window of the last 20 "
        "message-target-success interactions"
    )
    assert set(audit["paper_contract"]["conditions"]) == {
        "memory_only",
        "env_board",
        "scratchpad",
        "codebook",
        "codebook_meta",
    }
    assert audit["paper_contract"]["capacity_sweep"] == [
        4,
        8,
        9,
        16,
        25,
        27,
        64,
        125,
    ]
    replication = audit["reported_results"]["study2_replication_late_accuracy"]
    assert replication["scratchpad"]["capacity_25_n3"] == {
        "mean": 0.867,
        "std": 0.023,
    }
    assert replication["memory_only"]["capacity_64_n3"] == {
        "mean": 0.58,
        "std": 0.14,
    }
    assert audit["reported_results"]["study3_single_seed"][
        "capacity_64_memory_only_accuracy_by_window_5_10_20_40"
    ] == [0.5, 0.34, 0.52, 0.52]
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    decision = audit["decision"]
    assert decision["retain_evaluation_contracts"] == [
        "post_gate_architecture_capacity_history_sensitivity_factorial",
        "combined_memory_conflict_staleness_and_revision_trace",
        "task_utility_separate_from_representation_structure",
    ]
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["implementation_status"] == "evaluation_contract_only"
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_hierarchical_memory_theory_splits_design_axes_without_fake_plugin() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(HIERARCHICAL_MEMORY_THEORY_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["official_pdf_sha256"] == (
        "f356654da20624a64289315a2aadde151092d6f3c5e95caa2bb35bb080ffb5f8"
    )
    assert audit["paper"]["official_source_archive_sha256"] == (
        "07f45536fd90293069ea55bc045d50e27253e8e711013aa197a777c391df1b66"
    )
    assert audit["paper_contract"]["traversal_patterns"] == [
        "top_down_refinement",
        "collapsed_search_across_levels",
        "multi_view_parallel_retrieval",
        "reasoning_based_navigation",
    ]
    assert audit["empirical_evidence"] == {
        "new_controlled_experiments": 0,
        "systems_mapped_to_framework": 11,
        "coarsening_traversal_matched_factorial": False,
        "representative_self_sufficiency_calibration": False,
        "robot_policy_evaluation": False,
        "paper_statement": (
            "existing evidence is consistent with coarsening-traversal coupling, "
            "but a controlled comparison across the spectrum remains open"
        ),
    }
    assert set(audit["current_harness_mapping"]["fused_hierarchical_components"]) == {
        "AdjacentMergeStore",
        "TieredChunkMeanStore",
        "BoundaryChunkRetriever",
    }
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    decision = audit["decision"]
    assert decision["retain_interface_design_contracts"] == [
        "hierarchical_memory_partition_representative_traversal_decomposition",
        "calibrated_representative_self_sufficiency_traversal_compatibility",
        "partition_coherence_and_relevance_monotonicity_diagnostic",
    ]
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["implementation_status"] == (
        "post_active_chain_interface_design_contract"
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_alma_is_a_search_baseline_not_a_fake_policy_memory_plugin() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(ALMA_AUDIT.read_text(encoding="utf-8"))

    assert audit["official_source"]["commit"] == (
        "7f78ac89da5ca1f0d2ada635be93e0ba8fdf2c54"
    )
    assert audit["official_source"]["license"] == "Apache-2.0"
    assert audit["paper_method_contract"]["discovered_designs_per_benchmark"] == 43
    assert audit["paper_method_contract"]["parent_sampling"] == {
        "normalized_performance": "sigmoid(candidate_success - no_memory_success)",
        "score": "normalized_performance - alpha * log(1 + visit_count)",
        "alpha": 0.5,
        "temperature": 0.5,
        "replacement": False,
        "all_candidates_retain_nonzero_probability": True,
    }
    assert audit["reported_results_percent"]["gpt5_nano_overall"]["alma"] == 12.3
    assert audit["reported_results_percent"]["gpt5_mini_overall"]["alma"] == 53.9
    assert audit["official_result_artifacts"]["total_scored_design_records"] == 215
    assert audit["official_result_artifacts"]["scored_design_syntax_errors"] == 0
    assert (
        audit["official_result_artifacts"]["per_episode_evaluation_logs_present"]
        is False
    )
    retained = audit["retained_nonduplicate_search_mechanisms"]
    assert [item["name"] for item in retained] == [
        "content_addressed_visit_penalized_open_archive_sampler",
        "parent_code_stratified_success_failure_delta_reflection",
        "typed_sandboxed_generate_validate_debug_loop",
    ]
    assert retained[0]["implementation"] == "memory_harness.search_archive"
    decision = audit["decision"]
    assert decision["implementation_status"] == (
        "search_strategy_baseline_only_pre_gate_inactive"
    )
    assert decision["activation_gate"] == "fixed_memory_utility_gate"
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_memevolve_only_adds_nonduplicate_search_and_transfer_contracts() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(MEMEVOLVE_AUDIT.read_text(encoding="utf-8"))

    assert audit["official_release"]["commit"] == (
        "6035d5659d7a092dbfa6a87b1a32a3cee652ba54"
    )
    assert audit["official_release"]["compileall"] == "passed"
    assert audit["official_release"]["project_native_tests"] == 0
    reported = audit["paper"]["reported_flash_searcher_percent"]
    assert reported["gaia"]["delta_percentage_points"] == 4.24
    assert reported["xbench"]["delta_percentage_points"] == 5.0
    assert reported["webwalkerqa"]["delta_percentage_points"] == 3.53
    assert audit["official_release"]["paper_named_riva_provider_released"] is False
    retained = audit["retained_nonduplicate_mechanisms"]
    assert [mechanism["name"] for mechanism in retained] == [
        "performance_cost_latency_pareto_survivor_selector",
        "frozen_architecture_cross_task_transfer_confirmation",
    ]
    assert retained[0]["implementation"] == (
        "memory_harness.search_archive.ParetoRecord/pareto_ranks/"
        "select_pareto_survivors"
    )
    decision = audit["decision"]
    assert decision["worth_retaining"] is True
    assert decision["add_policy_memory_payload"] is False
    assert decision["add_to_source_audited_candidate_catalog"] is False
    assert decision["add_to_fixed_executable_suite"] is False
    assert decision["activate_search_before_fixed_memory_gate_1"] is False
    assert len(catalog["candidates"]) == 17


def test_muvla_candidate_preserves_training_and_inference_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item for item in payload["candidates"] if item["id"] == "muvla_recurrent_tokens"
    )

    assert {
        "episode_ordered_is_first_aware_dataloader",
        "memory_action_attention_guard",
        "matched_tbptt_k1_k2_training",
        "per_environment_step_recurrent_update",
    } <= set(candidate["requirements"])
    assert {
        "in_backbone_recurrent_token_writer",
        "memory_action_leakage_guard",
        "tbptt_recurrent_credit_assignment",
    } <= {operator["name"] for operator in candidate["operators"]}


def test_gmp_candidate_is_a_trained_read_controller_not_a_fake_fixed_switch() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "gmp_error_calibrated_read_gate"
    )

    assert candidate["entry_gate"] == "fixed_memory_controller_gate"
    assert {
        "winning_fixed_pi05_memory_program",
        "budget_matched_all_on_and_all_off_policies",
        "frozen_held_out_gate_calibration_split",
        "frozen_gate_then_full_policy_retraining",
        "matched_all_off_all_on_learned_oracle_gate_ablation",
    } <= set(candidate["requirements"])
    assert {
        "error_calibrated_memory_read_gate",
        "paired_policy_error_labeler",
        "frozen_read_gate_policy_retraining",
    } <= {operator["name"] for operator in candidate["operators"]}


def test_tfp_candidate_keeps_time_update_utilization_and_training_separable() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item for item in payload["candidates"] if item["id"] == "tfp_ltc_action_adaln"
    )

    assert candidate["entry_gate"] == "fixed_episode_memory_utility_gate"
    assert {
        "real_timestamp_or_audited_monotonic_delta_t",
        "hidden_state_bank_checkpoint_and_resume",
        "matched_real_vs_fixed_delta_t_ablation",
        "matched_concat_vs_action_adaln_utilization_ablation",
        "budget_matched_pi05_joint_training",
    } <= set(candidate["requirements"])
    assert {
        "visual_proprio_belief_input_encoder",
        "elapsed_time_ltc_state_update",
        "action_head_adaln_belief_utilizer",
        "episode_boundary_belief_reset",
        "episode_ordered_tbptt_belief_training",
    } <= {operator["name"] for operator in candidate["operators"]}


def test_nativemem_candidate_preserves_two_stage_representation_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "nativemem_action_supervised_tokens"
    )

    assert candidate["entry_gate"] == "fixed_episode_memory_utility_gate"
    assert {
        "stage1_frozen_vla_action_supervision",
        "frame_view_token_cache_with_checkpoint_provenance",
        "stage2_frozen_tokenizer_and_full_pi05_task_finetuning",
        "per_episode_reset_and_deployment_matched_update_cadence",
        "budget_includes_stage1_cache_and_stage2",
    } <= set(candidate["requirements"])
    assert {
        "action_supervised_native_visual_token_encoder",
        "dense_frame_view_memory_writer",
        "bounded_native_visual_token_queue",
        "input_sequence_native_memory_utilizer",
        "frozen_vla_memory_tokenizer_pretraining",
        "cached_token_task_finetuning",
    } <= {operator["name"] for operator in candidate["operators"]}


def test_optimusvla_candidate_preserves_action_prior_and_leakage_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "optimusvla_cross_episode_action_prior"
    )

    assert candidate["payload_family"] == "trajectory_action_prior"
    assert candidate["entry_gate"] == "fixed_episode_memory_utility_gate"
    assert {
        "exclude_eval_episodes_from_bank",
        "deployment_observable_progress_signal",
        "action_normalization_contract",
        "matched_gaussian_random_shuffled_wrong_task_prior_ablation",
        "matched_fixed_vs_confidence_adaptive_nfe",
        "budget_matched_pi05_policy_checkpoint",
    } <= set(candidate["requirements"])
    assert {
        "pooled_prefix_task_embedding_encoder",
        "task_contrastive_prior_head_training",
        "cross_episode_action_trajectory_bank",
        "faiss_semantic_trajectory_retriever",
        "progress_aligned_action_block_selector",
        "weighted_action_prior_flow_initializer",
        "confidence_adaptive_nfe_controller",
        "episode_action_prior_session_reset",
    } <= {operator["name"] for operator in candidate["operators"]}


def test_robomme_candidate_preserves_perceptual_factorial_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "robomme_perceptual_patch_memory"
    )

    assert candidate["payload_family"] == "temporal_visual_patch_memory"
    assert candidate["entry_gate"] == "fixed_episode_memory_utility_gate"
    assert {
        "preserve_frame_view_patch_and_position_identity",
        "causal_no_future_history_index_audit",
        "training_inference_prefix_causal_selector_parity",
        "released_full_episode_heap_compatibility_ablation",
        "right_padding_and_validity_mask_contract",
        "released_six_perceptual_checkpoint_hash_verification",
        "matched_framesamp_tokendrop_by_context_modulator_expert_factorial",
        "matched_none_zero_and_shuffled_history_ablations",
        "matched_64_128_256_512_token_budget_curve",
        "locked_robomme_reproduction_before_rmbench_port",
    } <= set(candidate["requirements"])
    assert {
        "frozen_pi05_patch_memory_encoder",
        "causal_patch_feature_writer",
        "episode_patch_feature_bank",
        "uniform_frame_patch_budget_retriever",
        "prefix_causal_rgb_change_patch_budget_retriever",
        "temporal_patch_token_projector",
        "input_context_patch_utilizer",
        "action_cross_attention_adaln_modulator",
        "separate_memory_expert_utilizer",
        "episode_patch_bank_reset",
        "joint_perceptual_memory_pi05_training",
    } <= {operator["name"] for operator in candidate["operators"]}

    audit = json.loads(
        (PROJECT_ROOT / candidate["source_audit"]).read_text(encoding="utf-8")
    )
    assert audit["official_release"]["commit"] == (
        "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
    )
    assert audit["official_release"]["license_spdx"] == "Apache-2.0"
    checkpoints = audit["official_release"]["released_perceptual_checkpoints"]
    assert len(checkpoints) == 6
    assert {checkpoint["path"] for checkpoint in checkpoints} == {
        f"perceptual-{representation}-{utilizer}/79999.zip"
        for representation in ("framesamp", "tokendrop")
        for utilizer in ("context", "expert", "modul")
    }
    assert next(
        checkpoint
        for checkpoint in checkpoints
        if checkpoint["path"] == "perceptual-framesamp-modul/79999.zip"
    ) == {
        "path": "perceptual-framesamp-modul/79999.zip",
        "size_bytes": 11878954899,
        "sha256": ("86ef2b63d9b8ff4d3ea2e8d06826c45cb126834f176b94defe235acabb649f61"),
    }
    assert audit["source_smoke"]["status"] == "passed"
    assert audit["source_smoke"]["checks"]["raw_memory_feature_shape"] == [
        512,
        2048,
    ]
    assert audit["source_smoke"]["checks"]["projected_memory_token_shape"] == [
        1,
        512,
        1024,
    ]
    assert (
        audit["source_smoke"]["checks"]["tokendrop_retained_first_frame_patches"] == 64
    )
    assert audit["source_smoke"]["checks"]["tokendrop_retained_change_steps"] == [
        7,
        15,
        23,
        31,
        39,
        47,
        55,
        63,
    ]
    profile = audit["rmbench_selector_profile"]
    profile_path = PROJECT_ROOT / profile["artifact"]
    assert (
        hashlib.sha256(profile_path.read_bytes()).hexdigest()
        == profile["artifact_sha256"]
    )
    profile_report = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile_report["inputs"]["total_frames"] == 17612
    assert profile_report["decision"]["selectors_behaviorally_distinct_on_rmbench"]
    assert profile_report["decision"]["offline_online_membership_mismatch_confirmed"]
    assert profile["source_equivalence"]["official_random_image_exact_patch_identity"]
    assert (
        profile["selector_distinctness"]["framesamp_tokendrop_exact_frame_set_fraction"]
        < 0.01
    )
    assert (
        profile["training_deployment_parity"][
            "released_offline_online_exact_token_parity_fraction"
        ]
        < 1.0
    )
    assert profile["training_deployment_parity"]["future_tokens_returned"] == 0
    assert audit["reported_results"]["framesamp_modulator_absolute_gain_over_pi05"] == (
        0.2658
    )


def test_trace_candidate_preserves_address_content_and_training_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "trace_trajectory_routed_slots"
    )

    assert candidate["payload_family"] == "trajectory_addressed_evidence_slots"
    assert candidate["entry_gate"] == "fixed_episode_memory_utility_gate"
    assert {
        "true_order_sensitive_signatory_backend_fail_closed",
        "bounded_streaming_path_signature_state_or_explicit_address_budget",
        "deployment_observable_executed_robot_state_only",
        "causal_prefix_training_inference_address_parity",
        "write_current_evidence_before_policy_query",
        "matched_no_memory_unrouted_no_delta_mean_readout_and_shuffled_address_ablations",
        "budget_matched_pi05_ordered_prefix_training",
    } <= set(candidate["requirements"])
    assert {
        "streamed_executed_state_path_signature_encoder",
        "current_visual_state_evidence_encoder",
        "signature_routed_gated_slot_update",
        "trajectory_conditioned_slot_readout",
        "trace_policy_condition_adapter",
        "episode_boundary_trace_reset",
        "masked_causal_prefix_trace_training",
    } <= {operator["name"] for operator in candidate["operators"]}

    audit = json.loads(
        (PROJECT_ROOT / candidate["source_audit"]).read_text(encoding="utf-8")
    )
    assert audit["official_release"]["commit"] == (
        "45bfb5b0a29de5eb59683da407b3da233cf3bc10"
    )
    assert audit["official_release"]["repository_license_status"] == (
        "no_root_license_file"
    )
    assert audit["source_checks"]["simple_backend_order_test"]["outputs_exactly_equal"]
    assert audit["reported_results"]["same_regression_base_component_ablation"] == {
        "current_observation_only": 25.5,
        "signature_only": 45.5,
        "unrouted_slot_memory": 52.17,
        "no_delta_routing": 61.43,
        "mean_readout": 62.8,
        "no_auxiliary_losses": 66.1,
        "full_trace": 69.23,
    }
    assert audit["reported_results"]["scope_note"].endswith("not TRACE-on-pi0.5.")


def test_aha_wam_candidate_preserves_dual_rate_context_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "aha_wam_dual_rate_planner_context"
    )

    assert candidate["payload_family"] == ("asynchronous_layerwise_planner_kv_context")
    assert candidate["entry_gate"] == "fixed_episode_memory_utility_gate"
    assert {
        "official_checkpoint_config_reproduction_and_offset_ambiguity_resolution",
        "bounded_per_layer_fifo_with_exact_byte_budget",
        "causal_chunk_aligned_observation_training_without_future_leakage",
        "training_deployment_planner_executor_phase_parity",
        "explicit_context_version_age_generation_and_stale_error_contract",
        "matched_none_synchronous_no_fifo_no_ovcr_no_offset_and_stale_context_ablations",
        "source_checkpoint_reproduction_before_pi05_port",
    } <= set(candidate["requirements"])
    assert {
        "slow_video_planner_context_encoder",
        "bounded_layerwise_planner_kv_fifo",
        "current_observation_guided_context_router",
        "action_dit_joint_context_utilizer",
        "planner_context_version_age_guard",
        "asynchronous_planner_executor_refresh",
        "episode_boundary_planner_context_reset",
        "planner_executor_phase_offset_training",
    } <= {operator["name"] for operator in candidate["operators"]}

    audit = json.loads(
        (PROJECT_ROOT / candidate["source_audit"]).read_text(encoding="utf-8")
    )
    assert audit["official_release"]["commit"] == (
        "471b9815738471d6e1758111b73a5198553c1817"
    )
    assert audit["official_release"]["license_spdx"] == "MIT"
    assert audit["reported_results"]["robotwin_50_task_average_success_percent"] == {
        "fast_wam": 91.83,
        "naive_async": 88.6,
        "kv_memory_only": 91.01,
        "ovcr_only": 91.47,
        "aha_wam_full": 92.8,
    }
    assert audit["source_smoke"]["core_router"][
        "zero_initialized_editor_is_exact_identity"
    ]
    assert audit["source_smoke"]["versioned_state_buffer"][
        "clear_resets_state_version_and_age"
    ]
    assert not audit["source_smoke"]["single_thread_staleness"][
        "explicit_action_error_result_emitted"
    ]
    checkpoint = audit["official_release"]["released_checkpoint_repositories"][0]
    assert checkpoint["revision"] == "1e15150a40c4688c0e1a716420052e0ef3cc770b"
    full = next(
        item for item in checkpoint["files"] if item["path"] == "robotwin_ahawam.pt"
    )
    assert full == {
        "path": "robotwin_ahawam.pt",
        "size_bytes": 14699235129,
        "lfs_sha256": (
            "aef9b283da72e5cefa41ea3f40c7ec3c6f074b3b3f099d5ff51b5a709bde3491"
        ),
    }


def test_kcvla_candidate_preserves_event_write_and_phase_lifecycle_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "kcvla_task_phase_event_keyframes"
    )

    assert candidate["payload_family"] == "task_phase_keyframe_history"
    assert candidate["entry_gate"] == "fixed_episode_memory_utility_gate"
    assert {
        "recursive_prefix_phase_threshold_calibration_on_train_or_calibration_split",
        "explicit_provisional_and_confirmed_candidate_lifecycle",
        "phase_error_propagation_and_recovery_trace",
        "per_episode_pending_confirmed_phase_and_history_reset",
        "matched_current_only_fixed_stride_oracle_learned_shuffled_and_stale_ablations",
        "learned_writer_and_policy_training_provenance",
        "source_checkpoint_reproduction_before_pi05_port",
    } <= set(candidate["requirements"])
    assert {
        "stage1_phase_metric_visual_encoder",
        "task_phase_film_event_query_encoder",
        "greedy_temporal_candidate_latch",
        "verified_phase_transition_keyframe_writer",
        "bounded_multiview_state_keyframe_queue",
        "chronological_multiview_keyframe_prefix_utilizer",
        "episode_task_phase_keyframe_reset",
        "phase_metric_pretraining",
        "balanced_phase_event_detector_training",
        "keyframe_conditioned_policy_finetuning",
    } <= {operator["name"] for operator in candidate["operators"]}

    audit = json.loads(
        (PROJECT_ROOT / candidate["source_audit"]).read_text(encoding="utf-8")
    )
    assert audit["official_release"]["commit"] == (
        "c9991fc7b453e64859538122c1bb26176c0b0e03"
    )
    assert audit["official_release"]["repository_license_status"] == (
        "no_root_license_file"
    )
    assert audit["official_release"]["license_spdx"] is None
    checkpoint = audit["official_release"]["released_model_repository"]["files"][0]
    assert checkpoint == {
        "path": "KSM_model/best_model_stage2.pth",
        "size_bytes": 46104610,
        "lfs_sha256": (
            "89ffc3cb9831081b52ac0384545a9316c849d6e8e95fdb50bc9dcc1475707947"
        ),
        "downloaded_and_verified": True,
    }
    assert (
        audit["source_smoke"]["released_ksm_checkpoint"]["strict_load_missing_keys"]
        == []
    )
    assert (
        audit["source_smoke"]["released_ksm_checkpoint"]["strict_load_unexpected_keys"]
        == []
    )
    assert audit["source_smoke"]["released_ksm_checkpoint"]["outputs_finite"]
    assert audit["source_smoke"]["architecture"]["total_parameters"] == 11503937
    assert audit["reported_results"]["maniskill_success_percent"] == {
        "no_history_gr00t": {
            "spatial": 20,
            "temporal": 0,
            "identity": 28,
            "counting": 16,
            "average": 16,
        },
        "best_fixed_stride_i40": {
            "spatial": 60,
            "temporal": 84,
            "identity": 84,
            "counting": 0,
            "average": 57,
        },
        "kc_vla": {
            "spatial": 70,
            "temporal": 98,
            "identity": 100,
            "counting": 100,
            "average": 92,
        },
    }
    assert audit["paper_and_source_contract"]["read_policy"].endswith(
        "retains the latest queue_len=6 entries."
    )
    assert (
        "ground-truth keyframe indices"
        in audit["source_paper_mismatches_and_limits"][0]
    )
    assert audit["decision"]["executable_registry_status"].startswith("not_added")


def test_memoryvla_candidate_preserves_complete_dual_stream_contract() -> None:
    payload = validate_source_audited_candidates(CATALOG)
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "memoryvla_dual_stream_pcmb"
    )

    assert candidate["payload_family"] == ("dual_stream_perceptual_cognitive_memory")
    assert candidate["entry_gate"] == "fixed_episode_memory_utility_gate"
    assert {
        "official_checkpoint_hash_and_config_reproduction",
        "training_deployment_memory_update_cadence_parity",
        "duplicate_group_padding_validity_mask",
        "explicit_raw_vs_fused_write_and_merged_timestamp_policy",
        "matched_none_cognitive_perceptual_dual_factorial",
        "matched_no_retrieval_no_timestep_add_fifo_shuffled_stale_ablations",
        "budget_matched_pi05_joint_training",
    } <= set(candidate["requirements"])
    assert {
        "vlm_eos_cognitive_token_encoder",
        "dual_vision_patch_bottleneck_encoder",
        "independent_dual_stream_episode_store",
        "adjacent_similarity_dual_stream_consolidator",
        "timestep_conditioned_dual_stream_cross_retriever",
        "adaptive_dual_stream_gate_fusion",
        "cognitive_diffusion_condition_utilizer",
        "perceptual_action_cross_attention_utilizer",
        "episode_boundary_dual_bank_reset",
        "ordered_group_joint_memory_action_training",
    } <= {operator["name"] for operator in candidate["operators"]}

    audit = json.loads(
        (PROJECT_ROOT / candidate["source_audit"]).read_text(encoding="utf-8")
    )
    assert audit["official_release"]["commit"] == (
        "d732ea9072bc063399ccc817aed74ab172eb50be"
    )
    assert audit["official_release"]["license_spdx"] is None
    checkpoint = audit["official_release"]["released_checkpoint_repository"]
    assert checkpoint["revision"] == ("399d2130b1dd92612ddc8eddae9e4990ea15fe09")
    assert checkpoint["model_file"] == {
        "path": "checkpoints/memvla-libero-100.pt",
        "size_bytes": 33507487606,
        "lfs_sha256": (
            "9e95b80e1f804bc7bfb292b3a3682b244647d319fdd93f48cb08353e649fba1f"
        ),
    }
    assert (
        audit["official_release"]["released_checkpoint_config"]["update_fused"] is False
    )
    assert audit["reported_results"]["same_family_memory_type_ablation_percent"] == {
        "cognitive_only": 63.5,
        "perceptual_only": 64.6,
        "dual_stream": 71.9,
    }
    assert (
        audit["parameter_and_storage_contract"][
            "estimated_memory_architecture_delta_parameters"
        ]
        == 575800736
    )
    assert audit["source_smoke"]["memory_operator_smoke"][
        "released_default_raw_write_confirmed"
    ]
    assert audit["source_smoke"]["action_utilizer_smoke"][
        "changing_perceptual_tokens_is_exact_no_op_before_training"
    ]


def test_lamem_vla_stays_a_typed_paper_contract_until_source_exists() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(LAMEM_AUDIT.read_text(encoding="utf-8"))

    assert audit["public_release"]["source_available"] is False
    assert audit["public_release"]["checkpoint_runnable_from_public_materials"] is False
    assert audit["decision"]["implementation_status"] == "paper_contract_only"
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert audit["reported_ablations_percent"]["memory_stream_factorial"] == {
        "none": [57.3, 92.1],
        "long_only_remove_short": [65.6, 95.4],
        "short_only_remove_long": [64.6, 94.8],
        "dual": [73.9, 97.0],
    }
    operators = audit["typed_decomposition"]
    assert all(
        operator["role"]
        in {
            "encoder",
            "writer",
            "store",
            "retriever",
            "utilizer",
            "lifecycle",
            "controller",
            "training",
        }
        and operator["inputs"]
        and operator["output"]
        for operator in operators
    )
    assert {
        "action_hidden_long_semantic_unit_writer",
        "masked_multimodal_memory_query_builder",
        "dual_vault_cosine_topk_retriever",
        "query_conditioned_dual_latent_condenser",
        "source_tagged_vlm_prefix_weaver",
    } <= {operator["name"] for operator in operators}
    assert "lamem_vla_dual_latent_vault" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }


def test_vq_memory_stays_a_typed_paper_contract_until_source_exists() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(VQ_MEMORY_AUDIT.read_text(encoding="utf-8"))

    assert audit["public_release"]["source_available"] is False
    assert audit["public_release"]["checkpoint_available"] is False
    assert audit["decision"]["implementation_status"] == "paper_contract_only"
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert audit["decision"]["novel_payload_family"] == (
        "discrete_proprioceptive_phase_tokens"
    )
    assert audit["workspace_fit"]["deployment_signal_available"] is True
    operators = audit["typed_decomposition"]
    assert all(
        operator["role"]
        in {
            "encoder",
            "writer",
            "store",
            "retriever",
            "utilizer",
            "lifecycle",
            "controller",
            "training",
        }
        and operator["inputs"]
        and operator["output"]
        for operator in operators
    )
    assert {
        "causal_proprio_state_history_writer",
        "windowed_proprio_vqvae_encoder",
        "posthoc_codebook_cluster_mapper",
        "bounded_discrete_proprio_token_store",
        "dedicated_vocabulary_prefix_utilizer",
    } <= {operator["name"] for operator in operators}
    assert audit["reported_results_percent"]["single_task_pi0"][
        "rule_020_success_none_raw_vq"
    ] == [0.0, 0.0, 45.0]
    assert "vq_memory_discrete_proprio_history" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }


def test_hymes_stays_a_separate_typed_paper_reference_until_source_exists() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(HYMES_AUDIT.read_text(encoding="utf-8"))

    assert audit["public_release"]["source_available"] is False
    assert audit["public_release"]["corrected_12_task_protocol_available"] is False
    assert audit["decision"]["implementation_status"] == "paper_contract_only"
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert audit["decision"]["role_in_project"] == (
        "task_specific_executable_memory_strong_reference_and_agent_search_baseline"
    )
    assert audit["reported_results_percent"][
        "robomemarena_corrected_12_task_160_episode_overall"
    ] == {
        "pi05_csr_tsr": [52.5, 41.3],
        "predimem_csr_tsr": [61.7, 45.6],
        "hymes_csr_tsr": [66.2, 60.1],
    }
    assert audit["reported_results_percent"]["six_task_ablation_csr_tsr"] == {
        "one_shot_program_multimodal_pace": [63.3, 53.3],
        "refined_program_vision_only": [55.8, 50.0],
        "refined_program_proprioception_only": [63.0, 63.3],
        "refined_program_multimodal_pace": [71.8, 71.7],
    }
    operators = audit["typed_decomposition"]
    assert all(
        operator["role"]
        in {
            "encoder",
            "writer",
            "store",
            "retriever",
            "utilizer",
            "lifecycle",
            "controller",
            "training",
        }
        and operator["inputs"]
        and operator["output"]
        for operator in operators
    )
    assert {
        "executable_symbolic_task_state_store",
        "k_of_w_multimodal_progress_verifier",
        "verified_symbolic_state_transition_writer",
        "memory_conditioned_flow_gradient_steerer",
        "rollout_trace_heuristic_program_refiner",
        "development_best_program_freezer",
    } <= {operator["name"] for operator in operators}
    assert "hymes_executable_symbolic_memory" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }


def test_rtcf_extends_action_prior_operators_without_duplicate_payload_family() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(RTCF_AUDIT.read_text(encoding="utf-8"))

    assert audit["public_release"]["source_available"] is False
    assert audit["public_release"]["checkpoint_available"] is False
    assert audit["decision"]["implementation_status"] == "paper_contract_only"
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert audit["decision"]["novel_payload_family"] is None
    assert audit["decision"]["reuse_payload_family"] == "trajectory_action_prior"
    assert audit["decision"]["retain_new_operators"] == [
        "incremental_monotonic_alignment_frontier_retriever",
        "aligned_future_action_chunk_selector",
        "clipped_low_frequency_action_residual_utilizer",
    ]
    assert audit["reported_results_percent"]["matched_four_suite_success"] == {
        "pi_fast": {
            "long": 61.6,
            "spatial": 96.4,
            "object": 97.4,
            "goal": 90.0,
            "all": 86.4,
        },
        "frame_nearest_neighbor_without_history_alignment": {
            "long": 63.6,
            "spatial": 95.8,
            "object": 96.8,
            "goal": 88.6,
            "all": 86.2,
        },
        "time_domain_without_frequency_selection": {
            "long": 50.0,
            "spatial": 96.2,
            "object": 97.2,
            "goal": 87.0,
            "all": 82.6,
        },
        "rtcf": {
            "long": 68.6,
            "spatial": 97.4,
            "object": 97.8,
            "goal": 89.8,
            "all": 88.4,
        },
    }
    operators = audit["typed_decomposition"]
    assert all(
        operator["role"]
        in {
            "encoder",
            "writer",
            "store",
            "retriever",
            "utilizer",
            "lifecycle",
            "controller",
            "training",
        }
        and operator["inputs"]
        and operator["output"]
        for operator in operators
    )
    assert {
        "frozen_success_visual_action_trajectory_bank",
        "incremental_monotonic_alignment_frontier_retriever",
        "aligned_future_action_chunk_selector",
        "clipped_low_frequency_action_residual_utilizer",
        "episode_alignment_frontier_reset",
    } <= {operator["name"] for operator in operators}
    assert "rtcf_progress_aligned_frequency_correction" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }


def test_atlasvla_retains_only_novel_voxel_world_state_family() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(ATLASVLA_AUDIT.read_text(encoding="utf-8"))

    assert audit["public_release"]["source_available"] is False
    assert audit["public_release"]["checkpoint_available"] is False
    assert audit["decision"]["implementation_status"] == "paper_contract_only"
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert (
        audit["decision"]["novel_payload_family"]
        == "voxel_hashed_spatiotemporal_world_state"
    )
    assert audit["decision"]["retain_new_operators"] == [
        "calibrated_wrist_rgb_depth_backprojection_encoder",
        "world_token_spatiotemporal_position_encoder",
        "depth_confidence_voxel_fusion_writer",
        "anchored_sliding_voxel_world_state_store",
        "ego_guided_voxel_world_retriever",
        "sequential_ego_world_action_dit_utilizer",
    ]
    assert audit["reported_results_percent"]["matched_component_ablation"] == {
        "without_world_state_memory": [93.5, 54.0],
        "without_ego_working_memory": [95.0, 56.5],
        "full_atlasvla": [97.6, 69.5],
        "without_world_state_update": [94.6, 58.0],
        "without_spatial_position_encoding": [96.4, 67.5],
        "without_temporal_position_encoding": [96.8, 65.0],
        "without_world_state_conditioning": [95.2, 61.5],
    }
    assert (
        "correct_vs_perturbed_camera_extrinsics_and_depth_confidence"
        in audit["required_pi05_ablations"]
    )
    operators = audit["typed_decomposition"]
    assert all(
        operator["role"]
        in {
            "encoder",
            "writer",
            "store",
            "retriever",
            "utilizer",
            "lifecycle",
            "training",
        }
        and operator["inputs"]
        and operator["output"]
        for operator in operators
    )
    assert "atlasvla_voxel_world_state_memory" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }


def test_g05_mem_is_source_audited_short_video_family_not_fixed_plugin() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(G05_MEM_AUDIT.read_text(encoding="utf-8"))
    candidate = next(
        item
        for item in catalog["candidates"]
        if item["id"] == "g05_mem_temporal_spatial_video"
    )

    assert audit["official_release"]["released_multiframe_memory_code"] is True
    assert audit["source_smoke"]["outputs_finite"] is True
    assert (
        audit["source_smoke"]["changing_only_historical_input_changes_current_output"]
        is True
    )
    assert audit["reported_results"]["memory_attribution_status"] == "not_isolated"
    assert audit["public_configuration_audit"]["base_model_memory_disabled"] is True
    assert (
        audit["public_configuration_audit"][
            "checked_task_configs_with_obs_size_greater_than_one"
        ]
        == 0
    )
    assert audit["decision"]["add_to_source_audited_catalog"] is True
    assert audit["decision"]["executable_registry_status"].startswith("not_added")
    assert candidate["payload_family"] == "bounded_multiview_raw_frame_window"
    assert {
        "cadence_matched_multiview_frame_writer",
        "same_patch_causal_temporal_value_mixer",
        "current_frame_token_drop_compressor",
        "whole_history_dropout_joint_training",
    } <= {operator["name"] for operator in candidate["operators"]}
    assert candidate["executable_architecture_alias"] is None


def test_onevomemory_is_a_paper_only_learned_controller_baseline() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(ONEVOMEMORY_AUDIT.read_text(encoding="utf-8"))

    assert audit["public_release"]["source_available"] is False
    assert audit["public_release"]["checkpoint_available"] is False
    assert audit["reported_results_percent"]["attribution_status"] == (
        "joint_system_only"
    )
    assert audit["non_contract_commented_draft"]["status"] == (
        "excluded_from_reproduction_contract"
    )
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    assert (
        audit["distinctness_and_deduplication"]["architecture_evolution_status"]
        == "fixed_architecture_parameter_adaptation"
    )
    assert audit["decision"]["retain_as_learned_controller_baseline"] is True
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert {
        "action_conditioned_value_key_value_encoder",
        "value_guided_elite_experience_writer",
        "temporal_value_delta_transition_writer",
        "successful_failed_rollout_memory_module_update",
    } <= {operator["name"] for operator in audit["typed_decomposition"]}
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert "onevomemory_value_guided_hierarchical_memory" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }


def test_streaming_grpo_adds_training_credit_not_a_payload_family() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(STREAMING_GRPO_AUDIT.read_text(encoding="utf-8"))

    assert audit["publication_status"]["peer_reviewed_paper"] is False
    assert audit["official_source"]["license"] == "MIT"
    assert audit["official_source"]["source_tests"] == {
        "selected_cpu_tests_passed": 48,
        "selected_cpu_tests_failed": 0,
        "covered_contracts": [
            "persistent keyframe buffer and temporal clustering",
            "streaming rollout causal accumulation",
            "selection validation",
            "coordinate conversion",
            "degenerate group handling",
            "select/use prompt schemas",
            "snapshot rollout and environment snapshot lifecycle",
        ],
    }
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    assert audit["decision"]["add_to_source_audited_payload_catalog"] is False
    assert audit["decision"]["add_to_executable_fixed_suite"] is False
    assert audit["decision"]["retain_training_operator"] is True
    assert audit["decision"]["retained_operator"] == (
        "delayed_outcome_streaming_memory_write_policy_optimization"
    )
    assert {
        "joint_subtask_keyframe_nomination_controller",
        "vote_cluster_temporal_coverage_keyframe_store",
        "delayed_outcome_streaming_memory_write_policy_optimization",
    } <= {operator["name"] for operator in audit["typed_decomposition"]}
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["source_contract"]["episode_scaffolding"].startswith(
        "the simulator oracle executes press and put-down"
    )
    assert len(catalog["candidates"]) == 17


def test_rbvla_adds_predictive_belief_training_not_a_duplicate_family() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(RBVLA_AUDIT.read_text(encoding="utf-8"))

    assert audit["public_release"]["source_available"] is False
    assert audit["public_release"]["checkpoint_available"] is False
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    assert audit["distinctness_and_deduplication"]["reuse_payload_family"] == (
        "continuous_recurrent_belief_state"
    )
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert audit["decision"]["retain_update_and_training_operators"] == [
        "stochastic_action_conditioned_recursive_belief_update",
        "ema_multihorizon_stochastic_belief_pretraining",
        "inverse_dynamics_belief_grounding",
    ]
    assert audit["reported_results_percent"][
        "matched_component_ablation_two_object_pick_place"
    ] == [
        {
            "frame_encoder_targets": False,
            "stochastic_latent": False,
            "belief_to_policy": False,
            "success": 32.5,
        },
        {
            "frame_encoder_targets": False,
            "stochastic_latent": False,
            "belief_to_policy": True,
            "success": 57.5,
        },
        {
            "frame_encoder_targets": False,
            "stochastic_latent": True,
            "belief_to_policy": True,
            "success": 62.5,
        },
        {
            "frame_encoder_targets": True,
            "stochastic_latent": True,
            "belief_to_policy": True,
            "success": 77.5,
        },
    ]
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert "rbvla_action_grounded_recursive_belief" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_stemvla_adds_geometry_encoder_not_a_duplicate_video_family() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(STEMVLA_AUDIT.read_text(encoding="utf-8"))

    assert audit["public_release"]["title_claims_open_source"] is True
    assert audit["public_release"]["source_available"] is False
    assert audit["public_release"]["checkpoint_available"] is False
    assert (
        audit["paper_contract"]["historical_encoder"]["history_length_disclosed"]
        is False
    )
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    assert audit["distinctness_and_deduplication"]["reuse_payload_family"] == (
        "bounded_multiview_raw_frame_window"
    )
    assert audit["decision"]["add_to_executable_suite"] is False
    assert audit["decision"]["add_to_source_audited_catalog"] is False
    assert audit["decision"]["retain_encoder_and_training_operators"] == [
        "vggt_implicit_geometry_history_encoder",
        "videoformer_temporal_geometry_aggregator",
        "future_geometry_distillation_objective",
    ]
    assert audit["reported_results_percent"]["matched_component_ablation"] == {
        "without_future_3d_geometry": {
            "Long": 67.0,
            "Object": 78.0,
            "Spatial": 76.5,
            "Goal": 72.0,
        },
        "without_4d_history": {
            "Long": 83.5,
            "Object": 92.0,
            "Spatial": 91.5,
            "Goal": 90.5,
        },
        "full": {
            "Long": 86.0,
            "Object": 96.0,
            "Spatial": 96.0,
            "Goal": 92.0,
        },
    }
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert "stemvla_implicit_video_geometry_history" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_hime_adds_controller_and_lifecycle_not_a_duplicate_payload() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(HIME_AUDIT.read_text(encoding="utf-8"))

    assert audit["official_source"]["commit"] == (
        "261766e3b067db8cf92fe2d34cc7da32034d142a"
    )
    assert audit["official_source"]["top_level_license"] is None
    assert audit["official_source"]["source_tests_present"] is False
    assert audit["source_smoke"] == {
        "encoder": "deterministic three-dimensional audit stub",
        "crud_resume_fifo_smoke": "passed",
        "query_top_k_requested": 2,
        "query_top_k_returned": 7,
        "finding": (
            "the exact-tag branch returns every record in the inverted-index "
            "posting list and ignores top_k"
        ),
    }
    assert audit["source_contract"]["ablation_profiles"]["fifo"].endswith(
        "source max_records=20"
    )
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    retained = [
        "subtask_completion_sentry_planner_trigger",
        "planner_generated_multimodal_crud_lifecycle",
    ]
    assert (
        audit["distinctness_and_deduplication"]["retain_source_level_operators"]
        == retained
    )
    assert audit["decision"]["retain_source_level_operators"] == retained
    assert audit["decision"]["add_new_payload_family"] is False
    assert audit["decision"]["add_to_source_audited_payload_catalog"] is False
    assert audit["decision"]["add_to_executable_fixed_suite"] is False
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert "hime_agent_managed_multimodal_memory" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_himem_wam_retains_boundary_training_not_a_duplicate_payload() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(HIMEM_WAM_AUDIT.read_text(encoding="utf-8"))

    assert audit["official_source"]["commit"] == (
        "cca2178458003a4fe44e9b52f6e5496868a372a5"
    )
    assert audit["official_source"]["license_spdx"] == "MIT"
    assert audit["official_checkpoint"]["files"][0] == {
        "path": "himem-wam_libero.pt",
        "size_bytes": 12041735545,
        "lfs_sha256": (
            "a7551d2aa2e3d70d39fd458c5b6b7b42f02b9e1f57f9c0d2561a1bd4ee70ce96"
        ),
        "downloaded_and_loaded": False,
    }
    assert audit["reported_results_percent"]["memory_attribution_status"] == (
        "not_isolated"
    )
    assert audit["source_smoke"]["false_write_gate"] == {
        "steps": 2,
        "threshold": 0.5,
        "gate_values": [0.0, 0.49],
        "stored_entries_after_false_gate": 2,
        "stored_entry_l1": 0.0,
        "finding": (
            "ExternalMemoryBank.write appends a zero placeholder even when the "
            "write gate is false, consuming capacity and changing memory length "
            "instead of preserving M_t"
        ),
    }
    assert audit["public_source_paths"]["documented_checkpoint_evaluation"][
        "model_creation"
    ].endswith("never calls build_policy")
    retained = [
        "skill_boundary_event_token_encoder",
        "boundary_supervised_sparse_memory_write_gate",
        "teacher_forced_boundary_memory_warmup",
    ]
    assert (
        audit["distinctness_and_deduplication"]["retain_source_level_operators"]
        == retained
    )
    assert audit["decision"]["retain_source_level_operators"] == retained
    assert audit["decision"]["add_new_payload_family"] is False
    assert audit["decision"]["add_to_source_audited_payload_catalog"] is False
    assert audit["decision"]["add_to_executable_fixed_suite"] is False
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert "himem_wam_skill_boundary_gated_memory" not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_worldscape_retains_distinct_planner_trace_as_paper_only_family() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(WORLDSCAPE_POLICY2_AUDIT.read_text(encoding="utf-8"))

    release = audit["public_release"]
    assert release["code_repository_commit"] == (
        "ea4562d39df737e6cb6fdf15ff75202f1eace791"
    )
    assert release["code_repository_files"] == [".gitignore", "README.md"]
    assert release["checkpoint_repository_revision"] == (
        "b8afa2b03758fd64731c6780b325fc70229be8b6"
    )
    assert release["checkpoint_repository_files"] == [".gitattributes", "README.md"]
    assert release["source_available"] is False
    assert release["checkpoint_available"] is False

    ablation = audit["reported_results_percent"][
        "matched_component_ablation_robotwin2_clean_only_training"
    ]
    assert ablation == {
        "none": {"clean": 64.6, "randomized": 17.22, "average": 40.91},
        "short_term_visual_memory": {
            "clean": 66.92,
            "randomized": 22.42,
            "average": 44.67,
        },
        "short_plus_long_event_memory": {
            "clean": 68.49,
            "randomized": 24.01,
            "average": 46.25,
        },
        "short_plus_long_event_memory_plus_latent_subgoal_reasoning": {
            "clean": 69.74,
            "randomized": 26.03,
            "average": 47.89,
        },
    }
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] == (
        "planner_reasoning_trace_event_history"
    )
    decision = audit["decision"]
    assert decision["add_new_paper_only_payload_family"] is True
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["retain_new_payload_and_operators"] == [
        "planner_reasoning_trace_event_history",
        "planner_perception_reasoning_event_encoder",
        "per_event_gist_planning_token_condenser",
        "global_local_boundary_event_view_builder",
        "event_caption_semantic_forcing",
    ]
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_mem_world_retains_future_view_retrieval_without_duplicate_payload() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(MEM_WORLD_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2606.18960v2"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "5a1de8441d868e599fac896701c92656293b445790ddc31e0da64dfa588624b9"
    )
    release = audit["public_release"]
    assert release["paper_or_source_official_code_url"] is None
    assert release["official_code_repository_found"] is False
    assert release["official_checkpoint_found"] is False
    assert release["official_dataset_release_found"] is False

    results = audit["reported_results"]["matched_retrieval_ablation"]
    assert results["third_view"]["short_term"]["object_consistency"] == 0.526
    assert results["third_view"]["stride"]["object_consistency"] == 0.544
    assert results["third_view"]["w_vmem"]["object_consistency"] == 0.597
    assert results["wrist_view"]["short_term"]["object_consistency"] == 0.401
    assert results["wrist_view"]["stride"]["object_consistency"] == 0.463
    assert results["wrist_view"]["w_vmem"]["object_consistency"] == 0.502

    retained = [
        "planned_action_chunk_future_wrist_pose_encoder",
        "wrist_only_temporal_surfel_frame_writer",
        "future_view_visibility_relevance_recency_retriever",
        "temporal_nms_history_frame_selector",
    ]
    assert audit["distinctness_and_deduplication"]["novel_payload_family"] is None
    assert (
        audit["distinctness_and_deduplication"]["retain_paper_level_operators"]
        == retained
    )
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_memora_retains_typed_lifecycle_without_duplicate_payload() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(MEMORA_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2607.14252v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "22e7380863d0f83e9df0e165b1b32331403bb2783232bfc1b9ad583539bc49b2"
    )
    release = audit["public_release"]
    assert release["project_url_http_status"] == 404
    assert release["official_code_repository_found"] is False
    assert release["official_checkpoint_found"] is False
    assert release["official_benchmark_release_found"] is False

    eam = audit["reported_results"]["matched_eam_qa_gemma4_31b_percent"]
    assert eam["flat_raw"]["overall"] == 52.0
    assert eam["flat_after_online_editing"]["overall"] == 54.0
    assert eam["memora_episodic_without_consolidation"]["overall"] == 60.1
    assert eam["memora_full"]["overall"] == 74.5
    assert eam["full_minus_episodic"]["overall"] == 14.4

    planning = audit["reported_results"]["qwen36_35b_planning_rgp"]
    assert planning["memora_episodic"] == {
        "replay": 0.345,
        "generalize": 0.425,
    }
    assert planning["memora_full"] == {
        "replay": 0.338,
        "generalize": 0.450,
    }
    assert planning["full_minus_episodic"] == {
        "replay": -0.007,
        "generalize": 0.025,
    }

    retained = [
        "evidence_linked_entity_state_history_editor",
        "evidence_time_snapshot_rollback_retriever",
        "evidence_linked_cross_episode_regularities_consolidator",
        "procedural_grounding_typed_query_router",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_mimir_retains_fail_closed_grounding_without_duplicate_payload() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(MIMIR_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.04933v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "6476b7c10f990725665c80160156a91413f14acc589a4b002b71e9894e45165f"
    )
    release = audit["public_release"]
    assert release["official_code_repository_found"] is False
    assert release["official_checkpoint_found"] is False
    assert release["official_dataset_or_run_release_found"] is False

    ablation = audit["reported_results_percent"]["matched_component_removal"]
    assert ablation["qwen3_vl_8b"] == {
        "full_mimir": {
            "eb_alfred_sr": 57.0,
            "eb_alfred_gc": 66.7,
            "eb_habitat_sr": 65.0,
            "eb_habitat_gc": 73.8,
        },
        "without_world_memory": {
            "eb_alfred_sr": 52.0,
            "eb_alfred_gc": 63.2,
            "eb_habitat_sr": 12.5,
            "eb_habitat_gc": 34.8,
        },
        "without_task_memory": {
            "eb_alfred_sr": 34.0,
            "eb_alfred_gc": 51.5,
            "eb_habitat_sr": 54.0,
            "eb_habitat_gc": 65.9,
        },
    }
    assert ablation["qwen3_vl_32b"]["full_mimir"] == {
        "eb_alfred_sr": 68.0,
        "eb_alfred_gc": 73.0,
        "eb_habitat_sr": 71.5,
        "eb_habitat_gc": 78.6,
    }

    retained = [
        "feedback_supported_observation_action_world_state_writer",
        "postcondition_verified_task_progress_agenda_updater",
        "active_goal_bounded_world_hypothesis_retriever",
        "fail_closed_active_goal_world_evidence_grounder",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_gesto_retains_distinct_human_activity_hierarchy_as_paper_only() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(GESTO_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.10886v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "837e4fa1ce3341cce106323988370bf18c4ce8fc863fca637b3d40a7b2f95374"
    )
    release = audit["public_release"]
    assert release["official_code_repository_found"] is False
    assert release["official_checkpoint_found"] is False
    assert release["official_query_or_run_release_found"] is False

    results = audit["reported_results"]
    assert results["main_comparison"]["gesto_full"] == [
        0.71,
        0.75,
        0.70,
        0.73,
        0.75,
    ]
    assert results["matched_component_ablation"]["without_event_hierarchy"] == [
        0.52,
        0.71,
        0.40,
        0.53,
        0.65,
    ]
    assert results["matched_component_ablation"][
        "without_context_grounding_refinement"
    ] == [0.55, 0.69, 0.59, 0.65, 0.61]

    retained = [
        "rgbd_atomic_human_object_interaction_encoder",
        "persistent_object_interaction_geometric_semantic_grounding_writer",
        "complete_time_ordered_goal_event_partition_consolidator",
        "context_agreement_unlinked_interaction_refiner",
        "refined_grounding_event_membership_reassignment",
        "bidirectional_activity_spatial_relation_retriever",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] == (
        "grounded_hierarchical_human_activity_event_memory"
    )
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is True
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_sma_retains_visit_reliability_without_duplicate_procedure_payload() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(SMA_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.12743v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "9d756a0a8248c90a6ff9543428e60858c20f22ad4d57dece3eefb1822a4d3906"
    )
    assert audit["paper"]["executable_source_files_in_archive"] == 0
    release = audit["public_release"]
    assert release["project_code_status"] == "Coming Soon"
    assert release["official_code_repository_found"] is False
    assert release["official_memory_bank_or_run_release_found"] is False

    results = audit["reported_results_percent"]
    assert results["main_five_benchmark_macro_average"]["qwen36_27b"] == {
        "no_memory": 63.3,
        "strongest_non_sma": 68.1,
        "sma": 69.8,
    }
    assert results["matched_robospatial_component_ablation_qwen36_27b"] == {
        "full_sma": 68.5,
        "without_summary": 65.3,
        "without_transferable_lesson": 65.0,
        "without_semantic_filter": 62.7,
        "with_raw_model_output_added": 64.1,
        "reward_only_reflection": 63.0,
    }
    assert results["representative_cross_model_transfer_122b_to_27b"][
        "robospatial_no_memory_to_transferred"
    ] == [54.1, 63.5]

    retained = [
        "verifier_guided_rollout_to_transferable_lesson_reflector",
        "one_pass_procedure_memory_writer",
        "visit_evidence_shrunk_transfer_reliability_updater",
        "semantic_then_transfer_reliability_reranker",
        "read_only_deployment_memory_value_freeze",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert "never treat card TRS as causal proof" in decision["research_agent_route"]
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_r4dsg_retains_relative_transition_operators_without_duplicate_payload() -> (
    None
):
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(R4DSG_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.11017v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "e4e5fa09e52ac80a42f61cc6df3f4890d6c4c5821389832b3996868bdbba9442"
    )
    assert audit["paper"]["executable_source_files_in_archive"] == 0
    release = audit["public_release"]
    assert release["official_repository_license"] == "MIT"
    assert release["official_repository_file_count"] == 21
    assert release["official_repository_executable_method_file_count"] == 0
    assert release["paper_claims_released_day1_scene_graph_outputs"] is True
    assert release["released_scene_graph_outputs_found_in_official_repository"] is False
    assert release["official_memory_json_or_qa_run_found"] is False

    control = audit["reported_results_percent"][
        "matched_option_blind_transition_control"
    ]
    assert control == {
        "no_transition_object_relation": {"overall": 34.9, "when": 34.7},
        "r4dsg_retrieval_plus_no_why": {"overall": 37.3, "when": 43.1},
        "r4dsg_minus_no_transition": {"overall": 2.4, "when": 8.4},
    }
    assert audit["reported_results_percent"]["memory_granularity_comparison"][
        "retrieval_plus"
    ] == {"overall": 39.6, "when": 43.1}

    retained = [
        "conservative_relative_object_identity_associator",
        "static_anchor_dynamic_object_role_inferencer",
        "anchor_relative_object_state_encoder",
        "persistent_anchor_change_event_writer",
        "segment_window_relational_memory_document_writer",
        "option_blind_object_transition_retriever",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_dreamfly_retains_visual_landmark_slots_as_trained_paper_contract() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(DREAMFLY_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.12308v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "8aa3f9d47571adebac39ab294deecdd0615cc65170c4b659040a15f802bc946b"
    )
    assert audit["paper"]["executable_source_files_in_archive"] == 0
    release = audit["public_release"]
    assert release["official_code_repository_found"] is False
    assert release["official_checkpoint_found"] is False
    assert release["official_memory_bank_or_run_release_found"] is False
    assert audit["terminology_boundary"]["not_claimed"].startswith("causal inference")

    progressive = audit["reported_results"]["progressive_memory_only"]
    assert progressive["dream_vla_baseline"] == {
        "ne_m": 67.82,
        "sr_percent": 21.55,
        "osr_percent": 42.32,
        "spl_percent": 16.09,
    }
    assert progressive["plus_causally_aligned_memory"] == {
        "ne_m": 48.93,
        "sr_percent": 24.11,
        "osr_percent": 48.22,
        "spl_percent": 19.85,
    }
    assert progressive["memory_delta"] == {
        "ne_m": -18.89,
        "sr_pp": 2.56,
        "osr_pp": 5.9,
        "spl_pp": 3.76,
    }
    removal = audit["reported_results"]["full_system_leave_one_out"]
    assert removal["full_minus_without_memory"] == {
        "ne_m": -17.05,
        "sr_pp": 11.86,
        "osr_pp": 22.88,
        "spl_pp": 8.41,
    }

    retained = [
        "instruction_conditioned_dense_and_region_candidate_encoder",
        "cross_observation_visual_spatial_evidence_track_writer",
        "persistent_or_single_observation_promotion_controller",
        "anchor_prototype_landmark_slot_store",
        "decision_read_before_write_memory_boundary",
        "current_visual_query_all_valid_slots_gated_cross_attention_utilizer",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] == (
        "instruction_conditioned_evidence_promoted_visual_landmark_slots"
    )
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["add_new_paper_only_payload_family"] is True
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_streamflow_retains_nonduplicate_operators_without_video_payload_alias() -> (
    None
):
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(STREAMFLOW_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.10949v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "47f820d36ad778dbfbdb7b5de9ba22767737b8b4c082cfcb567e2db36e84acdf"
    )
    assert audit["paper"]["supplement_present_in_source_archive"] is True
    assert audit["paper"]["executable_source_files_in_archive"] == 0
    release = audit["public_release"]
    assert release["official_code_repository_found"] is False
    assert release["official_checkpoint_found"] is False
    assert release["official_training_data_or_run_release_found"] is False

    component = audit["reported_results_percent"]["matched_component_ablation"]
    assert component["full"] == {"rtvu": 81.55, "videomme_long": 62.11}
    assert component["without_mid_term"] == {
        "rtvu": 76.86,
        "videomme_long": 60.33,
    }
    assert component["without_long_term"] == {
        "rtvu": 80.18,
        "videomme_long": 51.67,
    }
    insertion = audit["reported_results_percent"]["matched_insertion_policy_rtvu"]
    assert insertion == {
        "vas_guided": 81.55,
        "budget_matched_delimiter": 81.32,
        "budget_matched_random": 81.06,
        "no_long_term_insertion": 80.18,
        "vas_minus_delimiter_pp": 0.23,
        "vas_minus_random_pp": 0.49,
    }

    retained = [
        "pre_encoder_reference_residual_patch_selector",
        "adjacent_gop_reference_anchored_sparse_consolidator",
        "generation_prefix_max_frame_gop_retriever",
        "visual_attention_deficit_memory_read_controller",
        "generation_time_post_prefix_latent_injector",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_drivevla_m0_retains_labeled_case_ttt_without_executable_stub() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(DRIVEVLA_M0_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.10413v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "52591e7f3b634dd4cf522aa633e0a0453e07717532794e122a84bd7a59893838"
    )
    source = audit["official_source"]
    assert source["commit"] == "0a409e93f651643c2e5e7b55b4fdc5c3fcff19ca"
    assert source["license_spdx"] == "Apache-2.0"
    assert source["memory_named_tracked_files"] == 0
    assert source["implementation_status"] == (
        "base_and_retrieve_models_only_without_memory_generation_retrieval_or_ttt"
    )
    release = audit["official_checkpoint_release"]
    assert release["revision"] == "9260a4f8fdfca86a64cdc16af5e79a404bf78d2b"
    assert release["gated"] == "auto"
    assert release["memory_bank_found"] is False
    assert release["ttt_adapter_checkpoint_or_state_found"] is False

    retrieval = audit["reported_results"]["navsim_v1_pdms"][
        "same_base_retrieval_ablation"
    ]
    assert retrieval == {
        "base_without_memory": 91.0,
        "language_key": 90.7,
        "map_key": 91.7,
        "map_plus_agent_keys": 92.3,
    }
    injection = audit["reported_results"]["navsim_v1_pdms"][
        "knowledge_injection_ablation"
    ]
    assert injection == {
        "base_without_memory": 91.0,
        "offline_failure_lora_10_epochs": 91.2,
        "full_action_decoder_ttt": 92.4,
        "decoupled_lora_ttt": 92.3,
    }

    retained = [
        "offline_oracle_metric_failure_case_writer",
        "dual_static_dynamic_structural_key_encoder",
        "similarity_deduplicated_labeled_failure_case_store",
        "hierarchical_map_then_agent_structural_case_retriever",
        "similarity_triggered_test_time_adapter_controller",
        "per_scenario_adapter_reinitialize_lifecycle",
        "branch_decoupled_retrieved_supervision_lora_utilizer",
        "pathway_aware_static_dynamic_score_fuser",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] == (
        "retrieved_supervision_adaptation_case_memory"
    )
    assert distinctness["retain_paper_and_partial_source_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_and_partial_source_operators"] == retained
    assert decision["add_new_paper_only_payload_family"] is True
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_consolidator_retains_ltm_conditioned_routing_without_duplicate_payload() -> (
    None
):
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(CONSOLIDATOR_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.11701v1"
    source = audit["official_source"]
    assert source["commit"] == "7fe54e5d34ac9342f51029c8b0ddfb99fbd92a5e"
    assert source["license_spdx"] == "MIT"
    assert source["test_file_count"] == 0
    assert source["compileall"] == "passed"

    release = audit["official_checkpoint_release"]
    assert release["revision"] == "ef46f0a6514d8cef22f2bfa22b8913445b08888b"
    assert release["checkpoint_count"] == 31
    assert release["run_package_validation"].startswith(
        "ablation_results declares 30 of 30 runs complete"
    )

    results = audit["reported_results_percent"]
    assert results["consolidator_only_same_checkpoint"] == {
        "learned_ltm": {"mean": 87.02, "sample_sd": 1.76},
        "forced_identity_ltm": {"mean": 18.32, "sample_sd": 0.04},
        "learned_minus_identity_pp": {"mean": 68.70, "sample_sd": 1.76},
    }
    routing = results["direct_ltm_routing"]
    assert routing["routing_off_learned_ltm"]["mean"] == 44.38
    assert routing["routing_on_learned_ltm"]["mean"] == 87.02
    assert routing["paired_on_minus_off_pp"] == {
        "mean": 42.64,
        "sample_sd": 1.10,
    }
    assert routing["segment_2_immediate_stm_both_conditions"] == 89.90

    retained = [
        "slot_local_stm_to_ltm_revision_consolidator",
        "persistent_ltm_conditioned_same_level_slot_router",
        "boundary_kv_stm_clear_ltm_persist_lifecycle",
        "same_checkpoint_forced_identity_consolidation_intervention",
        "dual_immediate_stm_post_reset_ltm_training",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retain_source_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_source_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_streamttt_reuses_fast_weights_and_only_retains_update_lifecycle() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(STREAMTTT_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.13416v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "ef0d815bec0c61271bf6480035f22829f21e2795ea869dc95e1e4de3f605dcfe"
    )
    release = audit["release_search"]
    assert release["paper_statement"] == "Our code will be released."
    assert release["official_code_found"] is False
    assert release["official_checkpoint_found"] is False
    assert release["hugging_face_model_search"]["matching_release_count"] == 0

    results = audit["reported_results_percent"]
    architecture = results["same_backbone_architecture_and_data_ablation"]
    assert architecture["sliding_kv_only_joint"] == {
        "real_time_average": 68.19,
        "episodic_recall": 49.58,
        "two_metric_average": 58.89,
    }
    assert architecture["hybrid_kv_ttt_joint"] == {
        "real_time_average": 78.85,
        "episodic_recall": 59.55,
        "two_metric_average": 69.20,
    }
    assert architecture["hybrid_minus_sliding_pp"] == {
        "real_time_average": 10.66,
        "episodic_recall": 9.97,
        "two_metric_average": 10.31,
    }

    retained = [
        "input_dependent_momentum_decay_kv_binding_fast_weight_update",
        "forward_boundary_resumable_large_chunk_ttt_lifecycle",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert (
        "robo_ttt_fast_weight_state"
        in (distinctness["reuse_existing_payload_and_operator_families"])
    )
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_qcr_retains_target_bound_utilization_without_duplicate_procedure_store() -> (
    None
):
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(QCR_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2608.12847v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "aa3b6330dde07d02cdd0e788c5cbc7b35dd2bace161827146b76d18264281dc1"
    )
    release = audit["release_search"]
    assert release["official_code_found"] is False
    assert release["official_bank_or_benchmark_manifest_found"] is False
    assert release["official_raw_predictions_found"] is False

    results = audit["reported_results"]
    assert results["inventory"] == {
        "verified_source_trajectories": 623,
        "unique_target_instances": 2391,
        "seed_matched_runs_per_target_condition": 3,
        "environments": ["WebArena", "WorkArena", "AppWorld"],
    }
    end_to_end = results["end_to_end_percent"]
    assert end_to_end["full_trajectory"]["mean_success"] == 51.6
    assert end_to_end["qcr"]["mean_success"] == 62.3
    assert end_to_end["qcr_minus_full_trajectory_success_pp"] == 10.7
    assert end_to_end["qcr_online_token_reduction_vs_full_trajectory_percent"] == 48.9
    large_shift = results["large_binding_shift"]
    assert large_shift["stale_binding_error_percent"] == {
        "full_trajectory": 46.9,
        "qcr": 10.9,
    }
    assert large_shift["correct_rebinding_percent"] == {
        "full_trajectory": 31.7,
        "qcr": 77.8,
    }

    retained = [
        "query_conditioned_binding_safe_trajectory_reuse_adapter",
        "fixed_retrieval_post_selection_reuse_intervention",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_echovla_retains_spatial_revision_without_duplicate_dual_memory() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(ECHOVLA_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2511.18112v3"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "e401461da929c9b0a799e65c97642cab8ba748f53af52bf80ddd275e1095a7f7"
    )
    release = audit["release_search"]
    repository = release["title_matching_github_account"]
    assert repository["paper_linked_or_author_verified"] is False
    assert repository["candidate_commit"] == (
        "afbb4473a08d607f4db15cbf11715c5a34c6a294"
    )
    assert repository["tracked_file_count"] == 21
    assert repository["executable_model_or_data_files"] == 0
    assert release["official_code_found"] is False
    assert release["official_checkpoint_found"] is False
    assert release["hugging_face_model_search"]["matching_release_count"] == 0

    ablation = audit["reported_results_percent"][
        "same_model_component_ablation_pnp_counter_to_stove"
    ]
    assert ablation["full"] == {"mobile": 17, "static": 21}
    assert ablation["without_scene_memory"] == {"mobile": 9, "static": 16}
    assert ablation["scene_memory_absolute_effect_pp"] == {
        "mobile": 8,
        "static": 5,
    }
    assert audit["reported_results_percent"]["reported_negative_interference"] == {
        "task": "open refrigerator",
        "pi05": 50,
        "echovla": 40,
        "paper_explanation": (
            "dynamic occlusion and changing door geometry can create spatial "
            "misalignment or ghosting in the explicit scene map"
        ),
    }

    retained = [
        "reconstruction_discrepancy_voxel_write_gate",
        "coordinate_keyed_ema_voxel_revision_store",
        "local_frustum_spatial_feature_retriever",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retain_paper_level_operators"] == retained
    decision = audit["decision"]
    assert decision["retain_paper_level_operators"] == retained
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_catalog"] is False
    assert decision["add_to_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("none:")
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_decomposition"]
    )
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_dream_adds_pose_revision_lifecycle_without_duplicate_spatial_payload() -> None:
    catalog = validate_source_audited_candidates(CATALOG)
    audit = json.loads(DREAM_SPATIAL_AUDIT.read_text(encoding="utf-8"))

    assert audit["paper"]["arxiv_id"] == "2606.00576v1"
    assert audit["paper"]["manuscript_source_archive_sha256"] == (
        "b163c2bae4e00cd1642f83bdce493c84d3f9e0dfadb56fa5d88d99a5b3cce096"
    )
    assert audit["official_source"]["commit"] == (
        "6d558e25f045a0a414e2f7ebdfcab5b78940c83d"
    )
    assert audit["official_source"]["git_archive_sha256"] == (
        "be7c62d29edfc18e5abb714d20e08f072ba85772c9a4f7f8b2f13677dabc7267"
    )
    assert audit["official_source"]["source_test_file_count"] == 0
    assert audit["reported_results"]["aggregate_long_term_success"] == {
        "dynamem": {"successes": 39, "trials": 80, "percent": 48.8},
        "dream": {"successes": 50, "trials": 80, "percent": 62.5},
        "absolute_difference_pp": 13.7,
    }
    assert audit["reported_results"]["memory_specific_ablation_present"] is False

    retained = [
        "pose_graph_revision_reintegration_planner",
        "keyframe_aware_observation_archive_pruner",
        "keyframe_recent_semantic_feature_retention_tier",
        "spatial_revision_dependent_cache_invalidation",
    ]
    distinctness = audit["distinctness_and_deduplication"]
    assert distinctness["novel_payload_family"] is None
    assert distinctness["retained_nonduplicate_operators"] == retained
    assert all(
        operator["inputs"] and operator["output"]
        for operator in audit["typed_operator_decomposition"]
    )
    assert audit["local_implementation"]["module"] == (
        "memory_harness.spatial_reintegration"
    )
    decision = audit["decision"]
    assert decision["add_reusable_spatial_lifecycle_module"] is True
    assert decision["add_new_payload_family"] is False
    assert decision["add_to_source_audited_candidate_catalog"] is False
    assert decision["add_to_fixed_executable_suite"] is False
    assert decision["current_rmbench_route"].startswith("inactive:")
    assert audit["candidate_id"] not in {
        candidate["id"] for candidate in catalog["candidates"]
    }
    assert len(catalog["candidates"]) == 17


def test_source_audited_candidate_rejects_changed_audit_hash(
    tmp_path: pathlib.Path,
) -> None:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    project = tmp_path / "project"
    (project / "configs" / "architectures").mkdir(parents=True)
    (project / "artifacts").mkdir()
    for candidate in payload["candidates"]:
        source = PROJECT_ROOT / candidate["source_audit"]
        target = project / candidate["source_audit"]
        target.write_bytes(source.read_bytes())
    payload["candidates"][0]["source_audit_sha256"] = "0" * 64
    catalog = project / "configs" / "source_audited_candidates.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source audit hash changed"):
        validate_source_audited_candidates(catalog, project_root=project)


def test_source_audited_candidate_rejects_executable_claim(
    tmp_path: pathlib.Path,
) -> None:
    project = tmp_path / "project"
    (project / "configs" / "architectures").mkdir(parents=True)
    (project / "artifacts").mkdir()
    (project / "configs" / "architectures" / "fixed_anchor.json").write_text(
        "{}", encoding="utf-8"
    )
    audit = {
        "decision": {"executable_registry_status": "not_added"},
    }
    audit_path = project / "artifacts" / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "test",
        "candidates": [
            {
                "id": "candidate",
                "source_audit": "artifacts/audit.json",
                "source_audit_sha256": audit_hash,
                "payload_family": "test",
                "implementation_status": "not_executable",
                "operators": [
                    {
                        "role": "store",
                        "name": "candidate_store",
                        "inputs": ["item"],
                        "output": "bank",
                    }
                ],
                "requirements": ["training"],
                "entry_gate": "gate",
                "executable_architecture_alias": "anchor",
            }
        ],
    }
    catalog = project / "configs" / "source_audited_candidates.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot claim an executable alias"):
        validate_source_audited_candidates(catalog, project_root=project)


def test_source_audited_candidate_rejects_untyped_operator(
    tmp_path: pathlib.Path,
) -> None:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    project = tmp_path / "project"
    (project / "configs" / "architectures").mkdir(parents=True)
    (project / "artifacts").mkdir()
    for candidate in payload["candidates"]:
        source = PROJECT_ROOT / candidate["source_audit"]
        target = project / candidate["source_audit"]
        target.write_bytes(source.read_bytes())
    del payload["candidates"][0]["operators"][0]["output"]
    catalog = project / "configs" / "source_audited_candidates.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks a typed edge"):
        validate_source_audited_candidates(catalog, project_root=project)


def test_source_audited_candidate_rejects_duplicate_payload_family(
    tmp_path: pathlib.Path,
) -> None:
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    project = tmp_path / "project"
    (project / "configs" / "architectures").mkdir(parents=True)
    (project / "artifacts").mkdir()
    for candidate in payload["candidates"]:
        source = PROJECT_ROOT / candidate["source_audit"]
        target = project / candidate["source_audit"]
        target.write_bytes(source.read_bytes())
    payload["candidates"][1]["payload_family"] = payload["candidates"][0][
        "payload_family"
    ]
    catalog = project / "configs" / "source_audited_candidates.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source-audited payload family"):
        validate_source_audited_candidates(catalog, project_root=project)
