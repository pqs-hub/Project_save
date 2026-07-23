import pytest
import torch

from tpi_jepa.model import TPIWorldModel
from tpi_jepa.train import (
    configure_trainable_parameters,
    initialize_from_checkpoint,
    rollout_horizon_for_epoch,
    save_checkpoint,
    update_ema_if_encoder_trainable,
)


def test_rollout_curriculum_supports_coarse_long_horizon_steps() -> None:
    config = {
        "rollout_max_horizon": 32,
        "rollout_start_epoch": 1,
        "rollout_increase_every": 1,
        "rollout_start_horizon": 8,
        "rollout_horizon_increment": 8,
    }

    assert [rollout_horizon_for_epoch(epoch, config) for epoch in range(1, 6)] == [8, 16, 24, 32, 32]


def test_rollout_curriculum_supports_explicit_schedule() -> None:
    config = {
        "rollout_max_horizon": 32,
        "rollout_horizon_schedule": [1, 2, 4, 8, 16, 32, 32],
    }

    assert [rollout_horizon_for_epoch(epoch, config) for epoch in range(1, 9)] == [1, 2, 4, 8, 16, 32, 32, 32]


def test_rollout_curriculum_keeps_legacy_defaults() -> None:
    config = {
        "rollout_max_horizon": 5,
        "rollout_start_epoch": 2,
        "rollout_increase_every": 2,
    }

    assert [rollout_horizon_for_epoch(epoch, config) for epoch in range(1, 8)] == [1, 2, 2, 3, 3, 4, 4]


def _tiny_model() -> TPIWorldModel:
    return TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=4,
    )


def test_initialize_from_checkpoint_restores_weights(tmp_path) -> None:
    source = _tiny_model()
    config = {
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
    }
    checkpoint = tmp_path / "source.pt"
    save_checkpoint(checkpoint, source, config, feature_dim=12, relation_dim=4)
    target = _tiny_model()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()

    initialize_from_checkpoint(
        target,
        checkpoint,
        torch.device("cpu"),
        feature_dim=12,
        relation_dim=4,
    )

    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        assert torch.equal(source_parameter, target_parameter)


def test_initialize_from_checkpoint_rejects_feature_mismatch(tmp_path) -> None:
    source = _tiny_model()
    checkpoint = tmp_path / "source.pt"
    save_checkpoint(checkpoint, source, {}, feature_dim=12, relation_dim=4)

    with pytest.raises(ValueError, match="feature_dim"):
        initialize_from_checkpoint(
            _tiny_model(),
            checkpoint,
            torch.device("cpu"),
            feature_dim=13,
            relation_dim=4,
        )


def test_rollout_dynamics_mode_freezes_encoder_and_heads() -> None:
    model = _tiny_model()

    selected = configure_trainable_parameters(model, "rollout_dynamics")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert selected
    assert trainable_names
    assert all(name.startswith(("action_encoder.", "dynamics.")) for name in trainable_names)
    assert not model.online_encoder.input.weight.requires_grad
    assert not model.reward_head[0].weight.requires_grad


def test_trainable_parameter_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="trainable_modules"):
        configure_trainable_parameters(_tiny_model(), "heads_maybe")


def test_typed_utility_head_has_separate_scalar_and_sa_outputs() -> None:
    model = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=4,
        utility_head_type="typed_film",
    )
    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.randn(5, 4)

    out = model.predict_from_latent(z_state, 2, 1, relation)

    assert out["typed_marginal_pred"].shape == torch.Size([])
    assert out["typed_return_pred"].shape == torch.Size([])
    assert out["typed_sa_reduction_pred"].shape == torch.Size([2])
    loss = out["typed_marginal_pred"] + out["typed_return_pred"] + out["typed_sa_reduction_pred"].sum()
    loss.backward()
    assert model.typed_utility_head is not None
    assert model.typed_utility_head.type_film.weight.grad is not None


def test_typed_cone_utility_head_pools_relation_context() -> None:
    model = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=12,
        utility_head_type="typed_cone_film",
    )
    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[2, 0] = 1.0
    relation[:3, 4] = torch.tensor([0.25, 0.5, 1.0])
    relation[2:, 5] = torch.tensor([1.0, 0.5, 0.25])
    relation[:, 6] = 1.0

    out = model.predict_from_latent(z_state, 2, 1, relation)

    assert out["typed_marginal_pred"].shape == torch.Size([])
    assert out["typed_return_pred"].shape == torch.Size([])
    assert out["typed_sa_reduction_pred"].shape == torch.Size([2])
    loss = out["typed_marginal_pred"] + out["typed_return_pred"] + out["typed_sa_reduction_pred"].sum()
    loss.backward()
    assert model.typed_utility_head is None
    assert model.typed_cone_utility_head is not None
    assert model.typed_cone_utility_head.context.weight.grad is not None


def test_typed_cone_moe_starts_from_checkpoint_compatible_shared_predictions() -> None:
    kwargs = {
        "feature_dim": 12,
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
        "relation_dim": 12,
    }
    shared = TPIWorldModel(**kwargs, utility_head_type="typed_cone_film")
    moe = TPIWorldModel(**kwargs, utility_head_type="typed_cone_moe")
    incompatible = moe.load_state_dict(shared.state_dict(), strict=False)
    assert any("type_experts" in key for key in incompatible.missing_keys)

    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[:3, 4] = torch.tensor([0.25, 0.5, 1.0])
    relation[2:, 5] = torch.tensor([1.0, 0.5, 0.25])
    relation[:, 6] = 1.0
    shared.eval()
    moe.eval()
    shared_out = shared.predict_from_latent(z_state, 2, 1, relation)
    moe_out = moe.predict_from_latent(z_state, 2, 1, relation)

    assert torch.equal(shared_out["typed_marginal_pred"], moe_out["typed_marginal_pred"])
    assert torch.equal(shared_out["typed_return_pred"], moe_out["typed_return_pred"])
    assert torch.equal(shared_out["typed_sa_reduction_pred"], moe_out["typed_sa_reduction_pred"])

    loss = moe_out["typed_marginal_pred"] + moe_out["typed_return_pred"]
    loss.backward()
    assert moe.typed_cone_utility_head is not None
    assert moe.typed_cone_utility_head.type_experts[0][-1].weight.grad is not None


def test_horizon_moe_starts_exactly_from_typed_moe_and_receives_step_gradients() -> None:
    kwargs = {
        "feature_dim": 12,
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
        "relation_dim": 12,
    }
    moe = TPIWorldModel(**kwargs, utility_head_type="typed_cone_moe")
    horizon = TPIWorldModel(**kwargs, utility_head_type="typed_cone_horizon_moe")
    incompatible = horizon.load_state_dict(moe.state_dict(), strict=False)
    assert any("horizon_experts" in key for key in incompatible.missing_keys)

    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[:3, 4] = torch.tensor([0.25, 0.5, 1.0])
    relation[2:, 5] = torch.tensor([1.0, 0.5, 0.25])
    relation[:, 6] = 1.0
    moe.eval()
    horizon.eval()
    moe_out = moe.predict_from_latent(z_state, 2, 1, relation)
    horizon_out = horizon.predict_from_latent(
        z_state,
        2,
        1,
        relation,
        sequence_step=255,
    )

    assert torch.equal(moe_out["typed_marginal_pred"], horizon_out["typed_marginal_pred"])
    assert torch.equal(moe_out["typed_return_pred"], horizon_out["typed_return_pred"])
    assert torch.equal(moe_out["typed_sa_reduction_pred"], horizon_out["typed_sa_reduction_pred"])

    loss = horizon_out["typed_marginal_pred"] + horizon_out["typed_return_pred"]
    loss.backward()
    assert horizon.typed_cone_utility_head is not None
    assert horizon.typed_cone_utility_head.horizon_experts[0][-1].weight.grad is not None


def test_return_rank_moe_preserves_incumbent_outputs_and_isolates_adapter() -> None:
    kwargs = {
        "feature_dim": 12,
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
        "relation_dim": 12,
    }
    incumbent = TPIWorldModel(**kwargs, utility_head_type="typed_cone_moe")
    adapted = TPIWorldModel(**kwargs, utility_head_type="typed_cone_return_rank_moe")
    incompatible = adapted.load_state_dict(incumbent.state_dict(), strict=False)
    assert any("return_rank_experts" in key for key in incompatible.missing_keys)

    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[:3, 4] = torch.tensor([0.25, 0.5, 1.0])
    relation[2:, 5] = torch.tensor([1.0, 0.5, 0.25])
    relation[:, 6] = 1.0
    incumbent.eval()
    adapted.eval()
    incumbent_out = incumbent.predict_from_latent(z_state, 2, 1, relation)
    adapted_out = adapted.predict_from_latent(z_state, 2, 1, relation)

    assert torch.equal(incumbent_out["typed_marginal_pred"], adapted_out["typed_marginal_pred"])
    assert torch.equal(incumbent_out["typed_return_pred"], adapted_out["typed_return_pred"])
    assert torch.equal(
        incumbent_out["typed_sa_reduction_pred"],
        adapted_out["typed_sa_reduction_pred"],
    )

    selected = configure_trainable_parameters(adapted, "typed_return_rank_only")
    trainable_names = {name for name, parameter in adapted.named_parameters() if parameter.requires_grad}
    assert selected
    assert trainable_names
    assert all(
        name.startswith("typed_cone_utility_head.return_rank_experts.")
        for name in trainable_names
    )
    adapted_out["typed_return_pred"].backward()
    assert adapted.typed_cone_utility_head is not None
    assert adapted.typed_cone_utility_head.return_rank_experts[0][-1].weight.grad is not None
    assert adapted.typed_cone_utility_head.type_experts[0][-1].weight.grad is None


def test_return_horizon_rank_moe_preserves_incumbent_and_isolates_horizon_adapter() -> None:
    kwargs = {
        "feature_dim": 12,
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
        "relation_dim": 12,
    }
    incumbent = TPIWorldModel(**kwargs, utility_head_type="typed_cone_return_rank_moe")
    adapted = TPIWorldModel(
        **kwargs,
        utility_head_type="typed_cone_return_horizon_rank_moe",
    )
    incompatible = adapted.load_state_dict(incumbent.state_dict(), strict=False)
    assert any("horizon_return_rank_experts" in key for key in incompatible.missing_keys)

    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[:3, 4] = torch.tensor([0.25, 0.5, 1.0])
    relation[2:, 5] = torch.tensor([1.0, 0.5, 0.25])
    relation[:, 6] = 1.0
    incumbent.eval()
    adapted.eval()
    incumbent_out = incumbent.predict_from_latent(z_state, 2, 1, relation)
    adapted_out = adapted.predict_from_latent(
        z_state,
        2,
        1,
        relation,
        sequence_step=686,
    )

    assert torch.equal(incumbent_out["typed_marginal_pred"], adapted_out["typed_marginal_pred"])
    assert torch.equal(incumbent_out["typed_return_pred"], adapted_out["typed_return_pred"])
    assert torch.equal(
        incumbent_out["typed_sa_reduction_pred"],
        adapted_out["typed_sa_reduction_pred"],
    )

    selected = configure_trainable_parameters(adapted, "typed_return_horizon_only")
    trainable_names = {name for name, parameter in adapted.named_parameters() if parameter.requires_grad}
    assert selected
    assert trainable_names
    assert all(
        name.startswith("typed_cone_utility_head.horizon_return_rank_experts.")
        for name in trainable_names
    )
    adapted_out["typed_return_pred"].backward()
    assert adapted.typed_cone_utility_head is not None
    horizon_experts = adapted.typed_cone_utility_head.horizon_return_rank_experts
    assert horizon_experts[0][-1].weight.grad is not None
    assert adapted.typed_cone_utility_head.return_rank_experts[0][-1].weight.grad is None
    assert adapted.typed_cone_utility_head.type_experts[0][-1].weight.grad is None


def test_late_horizon_adapter_is_structurally_zero_through_b15() -> None:
    kwargs = {
        "feature_dim": 12,
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
        "relation_dim": 12,
    }
    incumbent = TPIWorldModel(
        **kwargs,
        utility_head_type="typed_cone_return_horizon_rank_moe",
    )
    adapted = TPIWorldModel(
        **kwargs,
        utility_head_type="typed_cone_return_late_horizon_rank_moe",
    )
    incompatible = adapted.load_state_dict(incumbent.state_dict(), strict=False)
    assert any("late_return_rank_experts" in key for key in incompatible.missing_keys)

    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[:3, 4] = torch.tensor([0.25, 0.5, 1.0])
    relation[2:, 5] = torch.tensor([1.0, 0.5, 0.25])
    relation[:, 6] = 1.0
    incumbent.eval()
    adapted.eval()

    with torch.no_grad():
        assert adapted.typed_cone_utility_head is not None
        for expert in adapted.typed_cone_utility_head.late_return_rank_experts:
            expert[-1].bias.fill_(0.25)

    incumbent_b15 = incumbent.predict_from_latent(
        z_state,
        2,
        1,
        relation,
        sequence_step=277,
    )
    adapted_b15 = adapted.predict_from_latent(
        z_state,
        2,
        1,
        relation,
        sequence_step=277,
    )
    incumbent_late = incumbent.predict_from_latent(
        z_state,
        2,
        1,
        relation,
        sequence_step=320,
    )
    adapted_late = adapted.predict_from_latent(
        z_state,
        2,
        1,
        relation,
        sequence_step=320,
    )

    assert torch.equal(incumbent_b15["typed_return_pred"], adapted_b15["typed_return_pred"])
    assert not torch.equal(incumbent_late["typed_return_pred"], adapted_late["typed_return_pred"])
    assert torch.equal(incumbent_late["typed_marginal_pred"], adapted_late["typed_marginal_pred"])
    assert torch.equal(
        incumbent_late["typed_sa_reduction_pred"],
        adapted_late["typed_sa_reduction_pred"],
    )

    selected = configure_trainable_parameters(adapted, "typed_return_late_horizon_only")
    trainable_names = {name for name, parameter in adapted.named_parameters() if parameter.requires_grad}
    assert selected
    assert trainable_names
    assert all(
        name.startswith("typed_cone_utility_head.late_return_rank_experts.")
        for name in trainable_names
    )
    adapted_late["typed_return_pred"].backward()
    late_experts = adapted.typed_cone_utility_head.late_return_rank_experts
    assert late_experts[0][-1].weight.grad is not None
    assert adapted.typed_cone_utility_head.horizon_return_rank_experts[0][-1].weight.grad is None
    assert adapted.typed_cone_utility_head.return_rank_experts[0][-1].weight.grad is None


def test_late_type_adapter_preserves_b15_and_within_type_ordering() -> None:
    kwargs = {
        "feature_dim": 12,
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
        "relation_dim": 12,
    }
    incumbent = TPIWorldModel(
        **kwargs,
        utility_head_type="typed_cone_return_horizon_rank_moe",
    )
    adapted = TPIWorldModel(
        **kwargs,
        utility_head_type="typed_cone_return_late_type_rank_moe",
    )
    incompatible = adapted.load_state_dict(incumbent.state_dict(), strict=False)
    assert any("late_type_calibrator" in key for key in incompatible.missing_keys)

    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[:3, 4] = torch.tensor([0.25, 0.5, 1.0])
    relation[2:, 5] = torch.tensor([1.0, 0.5, 0.25])
    relation[:, 6] = 1.0
    incumbent.eval()
    adapted.eval()
    assert adapted.typed_cone_utility_head is not None
    with torch.no_grad():
        adapted.typed_cone_utility_head.late_type_calibrator[-1].bias.fill_(0.25)

    incumbent_b15 = incumbent.predict_from_latent(z_state, 2, 1, relation, sequence_step=277)
    adapted_b15 = adapted.predict_from_latent(z_state, 2, 1, relation, sequence_step=277)
    assert torch.equal(incumbent_b15["typed_return_pred"], adapted_b15["typed_return_pred"])

    incumbent_late = incumbent.predict_from_latent(z_state, 2, 1, relation, sequence_step=320)
    adapted_late = adapted.predict_from_latent(z_state, 2, 1, relation, sequence_step=320)
    assert not torch.equal(incumbent_late["typed_return_pred"], adapted_late["typed_return_pred"])
    assert torch.equal(incumbent_late["typed_marginal_pred"], adapted_late["typed_marginal_pred"])
    assert torch.equal(
        incumbent_late["typed_sa_reduction_pred"],
        adapted_late["typed_sa_reduction_pred"],
    )

    incumbent_other = incumbent.predict_from_latent(z_state, 3, 1, relation, sequence_step=320)
    adapted_other = adapted.predict_from_latent(z_state, 3, 1, relation, sequence_step=320)
    first_shift = adapted_late["typed_return_pred"] - incumbent_late["typed_return_pred"]
    second_shift = adapted_other["typed_return_pred"] - incumbent_other["typed_return_pred"]
    assert torch.allclose(first_shift, second_shift)

    selected = configure_trainable_parameters(adapted, "typed_return_late_type_only")
    trainable_names = {name for name, parameter in adapted.named_parameters() if parameter.requires_grad}
    assert selected
    assert trainable_names
    assert all(
        name.startswith("typed_cone_utility_head.late_type_calibrator.")
        for name in trainable_names
    )
    adapted_late["typed_return_pred"].backward()
    assert adapted.typed_cone_utility_head.late_type_calibrator[-1].weight.grad is not None
    assert adapted.typed_cone_utility_head.horizon_return_rank_experts[0][-1].weight.grad is None
    assert adapted.typed_cone_utility_head.return_rank_experts[0][-1].weight.grad is None


def test_late_control_adapter_shares_cp0_cp1_shift() -> None:
    kwargs = {
        "feature_dim": 12,
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
        "relation_dim": 12,
    }
    incumbent = TPIWorldModel(
        **kwargs,
        utility_head_type="typed_cone_return_horizon_rank_moe",
    )
    adapted = TPIWorldModel(
        **kwargs,
        utility_head_type="typed_cone_return_late_control_rank_moe",
    )
    incompatible = adapted.load_state_dict(incumbent.state_dict(), strict=False)
    assert any("late_control_calibrator" in key for key in incompatible.missing_keys)

    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[:, 6] = 1.0
    incumbent.eval()
    adapted.eval()
    assert adapted.typed_cone_utility_head is not None
    with torch.no_grad():
        adapted.typed_cone_utility_head.late_control_calibrator[-1].bias.fill_(0.25)

    shifts = []
    for action_type_id in (0, 1, 2):
        incumbent_out = incumbent.predict_from_latent(
            z_state, 2, action_type_id, relation, sequence_step=320
        )
        adapted_out = adapted.predict_from_latent(
            z_state, 2, action_type_id, relation, sequence_step=320
        )
        shifts.append(adapted_out["typed_return_pred"] - incumbent_out["typed_return_pred"])
    assert torch.allclose(shifts[0], shifts[1])
    assert torch.allclose(shifts[0], torch.tensor(0.25))
    assert torch.equal(shifts[2], torch.tensor(0.0))

    incumbent_b15 = incumbent.predict_from_latent(z_state, 2, 0, relation, sequence_step=277)
    adapted_b15 = adapted.predict_from_latent(z_state, 2, 0, relation, sequence_step=277)
    assert torch.equal(incumbent_b15["typed_return_pred"], adapted_b15["typed_return_pred"])

    selected = configure_trainable_parameters(adapted, "typed_return_late_control_only")
    trainable_names = {name for name, parameter in adapted.named_parameters() if parameter.requires_grad}
    assert selected
    assert trainable_names
    assert all(
        name.startswith("typed_cone_utility_head.late_control_calibrator.")
        for name in trainable_names
    )
    adapted.predict_from_latent(
        z_state, 2, 0, relation, sequence_step=320
    )["typed_return_pred"].backward()
    assert adapted.typed_cone_utility_head.late_control_calibrator[-1].weight.grad is not None
    assert adapted.typed_cone_utility_head.horizon_return_rank_experts[0][-1].weight.grad is None


def test_marginal_horizon_rank_moe_preserves_other_voters_and_isolates_adapter() -> None:
    kwargs = {
        "feature_dim": 12,
        "latent_dim": 8,
        "encoder_layers": 1,
        "action_type_dim": 4,
        "dropout": 0.0,
        "relation_dim": 12,
    }
    incumbent = TPIWorldModel(**kwargs, utility_head_type="typed_cone_moe")
    adapted = TPIWorldModel(
        **kwargs,
        utility_head_type="typed_cone_marginal_horizon_rank_moe",
    )
    incompatible = adapted.load_state_dict(incumbent.state_dict(), strict=False)
    assert any("marginal_rank_experts" in key for key in incompatible.missing_keys)

    z_state = torch.randn(5, 8, requires_grad=True)
    relation = torch.zeros(5, 12)
    relation[:3, 4] = torch.tensor([0.25, 0.5, 1.0])
    relation[2:, 5] = torch.tensor([1.0, 0.5, 0.25])
    relation[:, 6] = 1.0
    incumbent.eval()
    adapted.eval()
    incumbent_out = incumbent.predict_from_latent(z_state, 2, 1, relation)
    adapted_out = adapted.predict_from_latent(z_state, 2, 1, relation, sequence_step=255)

    assert torch.equal(incumbent_out["typed_marginal_pred"], adapted_out["typed_marginal_pred"])
    assert torch.equal(incumbent_out["typed_return_pred"], adapted_out["typed_return_pred"])
    assert torch.equal(
        incumbent_out["typed_sa_reduction_pred"],
        adapted_out["typed_sa_reduction_pred"],
    )

    selected = configure_trainable_parameters(adapted, "typed_marginal_rank_only")
    trainable_names = {name for name, parameter in adapted.named_parameters() if parameter.requires_grad}
    assert selected
    assert trainable_names
    assert all(
        name.startswith("typed_cone_utility_head.marginal_rank_experts.")
        for name in trainable_names
    )
    adapted_out["typed_marginal_pred"].backward()
    assert adapted.typed_cone_utility_head is not None
    assert adapted.typed_cone_utility_head.marginal_rank_experts[0][-1].weight.grad is not None
    assert adapted.typed_cone_utility_head.type_experts[0][-1].weight.grad is None


def test_typed_cone_only_freezes_production_world_model() -> None:
    model = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=12,
        utility_head_type="typed_cone_film",
    )

    selected = configure_trainable_parameters(model, "typed_utility_only")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert selected
    assert trainable_names
    assert all(name.startswith("typed_cone_utility_head.") for name in trainable_names)
    assert not model.online_encoder.input.weight.requires_grad
    assert not model.q_head[0].weight.requires_grad


def test_typed_experts_only_preserves_shared_cone_head() -> None:
    model = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=12,
        utility_head_type="typed_cone_moe",
    )

    selected = configure_trainable_parameters(model, "typed_experts_only")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert selected
    assert trainable_names
    assert all(
        name.startswith("typed_cone_utility_head.expert_gate.")
        or name.startswith("typed_cone_utility_head.type_experts.")
        for name in trainable_names
    )
    assert not model.typed_cone_utility_head.context.weight.requires_grad
    assert not model.typed_cone_utility_head.marginal.weight.requires_grad


def test_typed_horizon_only_freezes_round8_moe_paths() -> None:
    model = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=12,
        utility_head_type="typed_cone_horizon_moe",
    )

    selected = configure_trainable_parameters(model, "typed_horizon_only")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert selected
    assert trainable_names
    assert all(
        name.startswith("typed_cone_utility_head.horizon_experts.")
        for name in trainable_names
    )
    assert not model.typed_cone_utility_head.context.weight.requires_grad
    assert not model.typed_cone_utility_head.type_experts[0][-1].weight.requires_grad


def test_utility_posttrain_freezes_encoder_but_trains_typed_heads() -> None:
    model = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=4,
        utility_head_type="typed_film",
    )

    selected = configure_trainable_parameters(model, "utility_posttrain")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert selected
    assert not model.online_encoder.input.weight.requires_grad
    assert any(name.startswith("typed_utility_head.") for name in trainable_names)
    assert model.action_encoder.type_emb.weight.requires_grad


def test_typed_utility_only_preserves_production_world_model() -> None:
    model = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=4,
        utility_head_type="typed_film",
    )

    selected = configure_trainable_parameters(model, "typed_utility_only")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert selected
    assert trainable_names
    assert all(name.startswith("typed_utility_head.") for name in trainable_names)
    assert not model.action_encoder.type_emb.weight.requires_grad
    assert not model.dynamics.mlp[0].weight.requires_grad
    assert not model.q_head[0].weight.requires_grad


def test_typed_utility_only_does_not_advance_target_ema() -> None:
    model = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=4,
        utility_head_type="typed_film",
    )
    configure_trainable_parameters(model, "typed_utility_only")
    with torch.no_grad():
        model.online_encoder.input.weight.add_(1.0)
    target_before = model.target_encoder.input.weight.detach().clone()

    update_ema_if_encoder_trainable(model, 0.5)

    assert torch.equal(model.target_encoder.input.weight, target_before)


def test_typed_model_can_initialize_from_legacy_checkpoint(tmp_path) -> None:
    source = _tiny_model()
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(checkpoint, source, {}, feature_dim=12, relation_dim=4)
    target = TPIWorldModel(
        feature_dim=12,
        latent_dim=8,
        encoder_layers=1,
        action_type_dim=4,
        dropout=0.0,
        relation_dim=4,
        utility_head_type="typed_film",
    )

    initialize_from_checkpoint(
        target,
        checkpoint,
        torch.device("cpu"),
        feature_dim=12,
        relation_dim=4,
        strict=False,
    )

    assert torch.equal(source.online_encoder.input.weight, target.online_encoder.input.weight)
