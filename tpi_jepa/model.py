"""Minimal PyTorch TPI-JEPA world model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .dataset import TPIDataset
from .graph import GATE_TYPES
from .labels import load_labels
from .testability import make_edge_weights


def mean_aggregate(
    num_nodes: int,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    h: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-aggregate source node states into destination nodes."""

    out = torch.zeros((num_nodes, h.shape[1]), dtype=h.dtype, device=h.device)
    deg = torch.zeros((num_nodes, 1), dtype=h.dtype, device=h.device)
    if edge_src.numel() == 0:
        return out
    if edge_weight is None:
        out.index_add_(0, edge_dst, h[edge_src])
        deg.index_add_(0, edge_dst, torch.ones((edge_dst.numel(), 1), dtype=h.dtype, device=h.device))
    else:
        weights = edge_weight.to(device=h.device, dtype=h.dtype).view(-1, 1)
        out.index_add_(0, edge_dst, h[edge_src] * weights)
        deg.index_add_(0, edge_dst, weights)
    return out / deg.clamp_min(1.0)


class NodeEncoder(nn.Module):
    """Encode node features with bidirectional directed message passing."""

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 64,
        layers: int = 3,
        dropout: float = 0.1,
        edge_weight_mode: str = "mean",
        edge_keep_ratio: float = 1.0,
        encoder_type: str = "mean",
    ):
        super().__init__()
        self.edge_weight_mode = edge_weight_mode
        self.edge_keep_ratio = edge_keep_ratio
        self.encoder_type = str(encoder_type or "mean").lower()
        self.input = nn.Linear(feature_dim, latent_dim)
        if self.encoder_type in {"gate_dir", "gate_directional", "topo_gate"}:
            self.gate_emb = nn.Embedding(len(GATE_TYPES), latent_dim)
            self.fanin_proj = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(layers)])
            self.fanout_proj = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(layers)])
            layer_input_dim = latent_dim * 4
        else:
            self.gate_emb = None
            self.fanin_proj = nn.ModuleList()
            self.fanout_proj = nn.ModuleList()
            layer_input_dim = latent_dim * 3
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layer_input_dim, latent_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim, latent_dim),
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(layers)])

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        gate_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return node latent vectors for one graph."""

        h = torch.relu(self.input(x))
        if self.gate_emb is not None and gate_type_ids is not None:
            h = h + self.gate_emb(gate_type_ids.to(device=x.device, dtype=torch.long))
        num_nodes = h.shape[0]
        edge_weight = make_edge_weights(x, edge_src, edge_dst, self.edge_weight_mode, self.edge_keep_ratio)
        for idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            fanin_msg = mean_aggregate(num_nodes, edge_src, edge_dst, h, edge_weight)
            fanout_msg = mean_aggregate(num_nodes, edge_dst, edge_src, h, edge_weight)
            if self.gate_emb is not None:
                fanin_msg = self.fanin_proj[idx](fanin_msg)
                fanout_msg = self.fanout_proj[idx](fanout_msg)
                gate_z = self.gate_emb(gate_type_ids.to(device=x.device, dtype=torch.long))
                update = layer(torch.cat([h, fanin_msg, fanout_msg, gate_z], dim=1))
            else:
                update = layer(torch.cat([h, fanin_msg, fanout_msg], dim=1))
            h = norm(h + update)
        return h


class ActionEncoder(nn.Module):
    """Encode an action from the selected node latent and action type."""

    def __init__(self, latent_dim: int = 64, action_type_dim: int = 16):
        super().__init__()
        self.type_emb = nn.Embedding(3, action_type_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_type_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, action_node_z: torch.Tensor, action_type_id: int | torch.Tensor) -> torch.Tensor:
        """Return one latent action embedding."""

        if not torch.is_tensor(action_type_id):
            action_type_id = torch.tensor(action_type_id, dtype=torch.long, device=action_node_z.device)
        action_type_id = action_type_id.to(device=action_node_z.device, dtype=torch.long).view(())
        type_z = self.type_emb(action_type_id)
        return self.mlp(torch.cat([action_node_z, type_z], dim=0))


class DynamicsPredictor(nn.Module):
    """Predict the next per-node latent state conditioned on one action."""

    def __init__(self, latent_dim: int = 64, relation_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2 + relation_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, action_emb: torch.Tensor, relation_features: torch.Tensor) -> torch.Tensor:
        """Return predicted next latent state for every node."""

        action_broadcast = action_emb.view(1, -1).expand(z.shape[0], -1)
        return self.mlp(torch.cat([z, action_broadcast, relation_features], dim=1))


class ResidualHardHead(nn.Module):
    """Node-level hard-fault head with optional action-cone context."""

    def __init__(
        self,
        latent_dim: int,
        relation_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        use_relation: bool = True,
    ):
        super().__init__()
        self.use_relation = bool(use_relation)
        input_dim = latent_dim + (relation_dim if self.use_relation else 0)
        self.input = nn.Linear(input_dim, latent_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.Sigmoid())
        self.output = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor, relation_features: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_relation:
            if relation_features is None:
                raise ValueError("ResidualHardHead with relation context requires relation_features")
            inputs = torch.cat([z, relation_features.to(device=z.device, dtype=z.dtype)], dim=1)
        else:
            inputs = z
        base = self.input(inputs)
        hidden = base + self.gate(inputs) * self.block(base)
        return self.output(hidden)


class TypedUtilityHead(nn.Module):
    """Action-conditioned marginal, long-return, and SA0/SA1 utility heads.

    The shared trunk sees the current action node, the predicted graph summary,
    and an explicit action-type embedding.  A FiLM transform makes the three
    action types use different affine views of the same structural context,
    while the output heads retain separately supervised semantics.
    """

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context = nn.Linear(summary_dim + latent_dim, latent_dim)
        self.type_film = nn.Linear(action_type_dim, latent_dim * 2)
        self.block = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.marginal = nn.Linear(latent_dim, 1)
        self.long_return = nn.Linear(latent_dim, 1)
        self.sa_reduction = nn.Linear(latent_dim, 2)

    def _hidden(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
    ) -> torch.Tensor:
        base = self.context(torch.cat([summary, action_node_z], dim=0))
        scale, bias = self.type_film(action_type_z).chunk(2, dim=0)
        conditioned = base * (1.0 + torch.tanh(scale)) + bias
        return base + self.block(conditioned)

    def _outputs(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "typed_marginal_pred": self.marginal(hidden).view(()),
            "typed_return_pred": self.long_return(hidden).view(()),
            "typed_sa_reduction_pred": self.sa_reduction(hidden),
        }

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._outputs(self._hidden(summary, action_node_z, action_type_z))


class TypedConeUtilityHead(TypedUtilityHead):
    """Typed utility head augmented with explicit action-cone state/effect pools."""

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__(summary_dim + cone_context_dim, latent_dim, action_type_dim, dropout)

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return super().forward(
            torch.cat([summary, cone_context], dim=0),
            action_node_z,
            action_type_z,
        )


class TypedConeMoEUtilityHead(TypedConeUtilityHead):
    """Cone utility head with zero-initialized type-gated residual experts.

    The inherited shared path is checkpoint-compatible with
    :class:`TypedConeUtilityHead`.  Three small experts specialize node ranking
    for the action types while a gate derived from the frozen type embedding
    mixes them.  Zero-initialized expert outputs make a migrated checkpoint
    start exactly from the shared-head predictions instead of perturbing a
    production policy before real-ATPG fine-tuning.
    """

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__(summary_dim, latent_dim, action_type_dim, cone_context_dim, dropout)
        expert_dim = max(8, latent_dim // 2)
        self.expert_gate = nn.Linear(action_type_dim, num_experts, bias=False)
        self.type_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, 4),
                )
                for _ in range(num_experts)
            ]
        )
        for expert in self.type_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self._hidden(
            torch.cat([summary, cone_context], dim=0),
            action_node_z,
            action_type_z,
        )
        outputs = self._outputs(hidden)
        gate = torch.softmax(self.expert_gate(action_type_z), dim=0)
        expert_values = torch.stack([expert(hidden) for expert in self.type_experts], dim=0)
        residual = (gate.view(-1, 1) * expert_values).sum(dim=0)
        return {
            "typed_marginal_pred": outputs["typed_marginal_pred"] + residual[0],
            "typed_return_pred": outputs["typed_return_pred"] + residual[1],
            "typed_sa_reduction_pred": outputs["typed_sa_reduction_pred"] + residual[2:4],
        }


class TypedConeReturnRankMoEUtilityHead(TypedConeMoEUtilityHead):
    """MoE utility head with an isolated return-ranking residual adapter.

    The existing shared and type-expert paths are checkpoint-compatible with
    :class:`TypedConeMoEUtilityHead`.  A second, zero-initialized expert bank
    can therefore learn oracle ordering for ``typed_return_pred`` without
    changing marginal-TC, SA-reduction, encoder, or dynamics predictions.
    """

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__(
            summary_dim,
            latent_dim,
            action_type_dim,
            cone_context_dim,
            dropout,
            num_experts,
        )
        expert_dim = max(8, latent_dim // 2)
        self.return_rank_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, 1),
                )
                for _ in range(num_experts)
            ]
        )
        for expert in self.return_rank_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self._hidden(
            torch.cat([summary, cone_context], dim=0),
            action_node_z,
            action_type_z,
        )
        outputs = self._outputs(hidden)
        gate = torch.softmax(self.expert_gate(action_type_z), dim=0)
        type_values = torch.stack([expert(hidden) for expert in self.type_experts], dim=0)
        type_residual = (gate.view(-1, 1) * type_values).sum(dim=0)
        return_values = torch.stack(
            [expert(hidden).reshape(()) for expert in self.return_rank_experts],
            dim=0,
        )
        return_residual = (gate * return_values).sum()
        return {
            "typed_marginal_pred": outputs["typed_marginal_pred"] + type_residual[0],
            "typed_return_pred": outputs["typed_return_pred"] + type_residual[1] + return_residual,
            "typed_sa_reduction_pred": outputs["typed_sa_reduction_pred"] + type_residual[2:4],
        }


class TypedConeHorizonMoEUtilityHead(TypedConeMoEUtilityHead):
    """MoE utility head with a zero-initialized sequence-position residual.

    Long planner rollouts previously forced the typed head to infer the
    sequence phase only from a recurrent latent that was trained mostly at
    prefixes no longer than 127.  Bounded analytic step features give a small
    expert branch direct access to the horizon while preserving an existing
    ``typed_cone_moe`` checkpoint exactly at initialization.
    """

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__(
            summary_dim,
            latent_dim,
            action_type_dim,
            cone_context_dim,
            dropout,
            num_experts,
        )
        self.horizon_feature_dim = 4
        expert_dim = max(8, latent_dim // 2)
        self.horizon_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim + self.horizon_feature_dim),
                    nn.Linear(latent_dim + self.horizon_feature_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, 4),
                )
                for _ in range(num_experts)
            ]
        )
        for expert in self.horizon_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    @staticmethod
    def _horizon_features(
        sequence_step: int | torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        step = torch.as_tensor(sequence_step, dtype=dtype, device=device).reshape(()).clamp_min(0.0)
        log_denominator = torch.log(step.new_tensor(1025.0))
        return torch.stack(
            [
                (torch.log1p(step) / log_denominator).clamp(0.0, 1.0),
                torch.exp(-step / 64.0),
                torch.exp(-step / 256.0),
                (1.0 - torch.exp(-step / 128.0)).clamp(0.0, 1.0),
            ]
        )

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
        sequence_step: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        hidden = self._hidden(
            torch.cat([summary, cone_context], dim=0),
            action_node_z,
            action_type_z,
        )
        outputs = self._outputs(hidden)
        gate = torch.softmax(self.expert_gate(action_type_z), dim=0)
        type_values = torch.stack([expert(hidden) for expert in self.type_experts], dim=0)
        type_residual = (gate.view(-1, 1) * type_values).sum(dim=0)

        horizon = self._horizon_features(
            sequence_step,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        horizon_input = torch.cat([hidden, horizon], dim=0)
        horizon_values = torch.stack(
            [expert(horizon_input) for expert in self.horizon_experts],
            dim=0,
        )
        horizon_residual = (gate.view(-1, 1) * horizon_values).sum(dim=0)
        residual = type_residual + horizon_residual
        return {
            "typed_marginal_pred": outputs["typed_marginal_pred"] + residual[0],
            "typed_return_pred": outputs["typed_return_pred"] + residual[1],
            "typed_sa_reduction_pred": outputs["typed_sa_reduction_pred"] + residual[2:4],
        }


class TypedConeReturnHorizonRankMoEUtilityHead(TypedConeReturnRankMoEUtilityHead):
    """Return ranker with an isolated explicit long-horizon residual.

    The inherited return adapter retains its learned node/type ordering.  A
    second zero-initialized expert bank receives bounded sequence-position
    features, so ultra-long ATPG labels can correct only the return score at
    late planner states without perturbing any incumbent voter at migration.
    """

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__(
            summary_dim,
            latent_dim,
            action_type_dim,
            cone_context_dim,
            dropout,
            num_experts,
        )
        self.horizon_feature_dim = 4
        expert_dim = max(8, latent_dim // 2)
        self.horizon_return_rank_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim + self.horizon_feature_dim),
                    nn.Linear(latent_dim + self.horizon_feature_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, 1),
                )
                for _ in range(num_experts)
            ]
        )
        for expert in self.horizon_return_rank_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
        sequence_step: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        hidden = self._hidden(
            torch.cat([summary, cone_context], dim=0),
            action_node_z,
            action_type_z,
        )
        outputs = self._outputs(hidden)
        gate = torch.softmax(self.expert_gate(action_type_z), dim=0)
        type_values = torch.stack([expert(hidden) for expert in self.type_experts], dim=0)
        type_residual = (gate.view(-1, 1) * type_values).sum(dim=0)
        return_values = torch.stack(
            [expert(hidden).reshape(()) for expert in self.return_rank_experts],
            dim=0,
        )
        return_residual = (gate * return_values).sum()

        horizon = TypedConeHorizonMoEUtilityHead._horizon_features(
            sequence_step,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        horizon_input = torch.cat([hidden, horizon], dim=0)
        horizon_values = torch.stack(
            [expert(horizon_input).reshape(()) for expert in self.horizon_return_rank_experts],
            dim=0,
        )
        horizon_residual = (gate * horizon_values).sum()
        return {
            "typed_marginal_pred": outputs["typed_marginal_pred"] + type_residual[0],
            "typed_return_pred": (
                outputs["typed_return_pred"]
                + type_residual[1]
                + return_residual
                + horizon_residual
            ),
            "typed_sa_reduction_pred": outputs["typed_sa_reduction_pred"] + type_residual[2:4],
        }


class TypedConeReturnLateHorizonRankMoEUtilityHead(TypedConeReturnHorizonRankMoEUtilityHead):
    """A return residual that is structurally inactive through the b15 budget.

    ``sequence_step`` is zero based in the planner, so the last action of the
    278-point b15 protocol is scored at step 277.  The late branch is exactly
    zero through that step and reaches full strength at the first ultra-long
    supervision prefix (320).  This lets long-horizon ATPG labels change
    b20/b21/b22/b17 behavior without perturbing the b15-selected incumbent.
    """

    late_start_step = 277.0
    late_full_step = 320.0

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__(
            summary_dim,
            latent_dim,
            action_type_dim,
            cone_context_dim,
            dropout,
            num_experts,
        )
        expert_dim = max(8, latent_dim // 2)
        self.late_return_rank_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim + self.horizon_feature_dim),
                    nn.Linear(latent_dim + self.horizon_feature_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, 1),
                )
                for _ in range(num_experts)
            ]
        )
        for expert in self.late_return_rank_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    @classmethod
    def _late_gate(
        cls,
        sequence_step: int | torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        step = torch.as_tensor(sequence_step, dtype=dtype, device=device).reshape(())
        width = cls.late_full_step - cls.late_start_step
        return ((step - cls.late_start_step) / width).clamp(0.0, 1.0)

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
        sequence_step: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(
            summary,
            action_node_z,
            action_type_z,
            cone_context,
            sequence_step=sequence_step,
        )
        hidden = self._hidden(
            torch.cat([summary, cone_context], dim=0),
            action_node_z,
            action_type_z,
        )
        gate = torch.softmax(self.expert_gate(action_type_z), dim=0)
        horizon = TypedConeHorizonMoEUtilityHead._horizon_features(
            sequence_step,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        horizon_input = torch.cat([hidden, horizon], dim=0)
        late_values = torch.stack(
            [expert(horizon_input).reshape(()) for expert in self.late_return_rank_experts],
            dim=0,
        )
        late_residual = (gate * late_values).sum()
        late_gate = self._late_gate(
            sequence_step,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return outputs | {
            "typed_return_pred": outputs["typed_return_pred"] + late_gate * late_residual,
        }


class TypedConeReturnLateTypeRankMoEUtilityHead(TypedConeReturnHorizonRankMoEUtilityHead):
    """Low-capacity late-horizon calibration of CP0/CP1/OP utility.

    The adapter sees only the frozen action-type embedding and analytic horizon
    features.  Consequently it can shift one action family relative to the
    others, but cannot relearn (and overfit) node ordering inside a family.
    Its last layer is zero initialized and the structural late gate is exactly
    zero through b15's final action, preserving the selected b15 policy.
    """

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__(
            summary_dim,
            latent_dim,
            action_type_dim,
            cone_context_dim,
            dropout,
            num_experts,
        )
        adapter_input_dim = action_type_dim + self.horizon_feature_dim
        adapter_hidden_dim = 8
        self.late_type_calibrator = nn.Sequential(
            nn.LayerNorm(adapter_input_dim),
            nn.Linear(adapter_input_dim, adapter_hidden_dim),
            nn.Tanh(),
            nn.Linear(adapter_hidden_dim, 1),
        )
        nn.init.zeros_(self.late_type_calibrator[-1].weight)
        nn.init.zeros_(self.late_type_calibrator[-1].bias)

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
        sequence_step: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(
            summary,
            action_node_z,
            action_type_z,
            cone_context,
            sequence_step=sequence_step,
        )
        horizon = TypedConeHorizonMoEUtilityHead._horizon_features(
            sequence_step,
            dtype=action_type_z.dtype,
            device=action_type_z.device,
        )
        type_residual = self.late_type_calibrator(
            torch.cat([action_type_z, horizon], dim=0)
        ).reshape(())
        late_gate = TypedConeReturnLateHorizonRankMoEUtilityHead._late_gate(
            sequence_step,
            dtype=action_type_z.dtype,
            device=action_type_z.device,
        )
        return outputs | {
            "typed_return_pred": outputs["typed_return_pred"] + late_gate * type_residual,
        }


class TypedConeReturnLateControlRankMoEUtilityHead(TypedConeReturnHorizonRankMoEUtilityHead):
    """Late binary Control-vs-Observe calibration with preserved polarity.

    Non-target oracle circuits agree much more strongly on whether a late test
    point should be a control point than on whether its forcing polarity should
    be CP0 or CP1.  This 25-parameter adapter therefore applies one identical
    horizon-dependent shift to both control types and leaves OP at zero.  It
    cannot alter node ordering or the incumbent CP0/CP1 preference.
    """

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__(
            summary_dim,
            latent_dim,
            action_type_dim,
            cone_context_dim,
            dropout,
            num_experts,
        )
        hidden_dim = 4
        self.late_control_calibrator = nn.Sequential(
            nn.Linear(self.horizon_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.late_control_calibrator[-1].weight)
        nn.init.zeros_(self.late_control_calibrator[-1].bias)

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
        sequence_step: int | torch.Tensor = 0,
        action_type_id: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(
            summary,
            action_node_z,
            action_type_z,
            cone_context,
            sequence_step=sequence_step,
        )
        horizon = TypedConeHorizonMoEUtilityHead._horizon_features(
            sequence_step,
            dtype=action_type_z.dtype,
            device=action_type_z.device,
        )
        control_residual = self.late_control_calibrator(horizon).reshape(())
        type_id = torch.as_tensor(
            action_type_id,
            dtype=torch.long,
            device=action_type_z.device,
        ).reshape(())
        # ActionEncoder uses 0=CP0, 1=CP1, 2=OP.
        is_control = (type_id != 2).to(dtype=action_type_z.dtype)
        late_gate = TypedConeReturnLateHorizonRankMoEUtilityHead._late_gate(
            sequence_step,
            dtype=action_type_z.dtype,
            device=action_type_z.device,
        )
        return outputs | {
            "typed_return_pred": (
                outputs["typed_return_pred"]
                + late_gate * is_control * control_residual
            ),
        }


class TypedConeMarginalHorizonRankMoEUtilityHead(TypedConeMoEUtilityHead):
    """Isolated horizon-aware oracle residual for marginal-TC ordering.

    Unlike the Round9 joint horizon experts, this branch cannot alter return,
    SA-reduction, shared typed utility, or legacy world-model predictions.  It
    adds one zero-initialized scalar to ``typed_marginal_pred`` and receives
    explicit sequence-position features for late-prefix supervision.
    """

    def __init__(
        self,
        summary_dim: int,
        latent_dim: int,
        action_type_dim: int,
        cone_context_dim: int,
        dropout: float = 0.1,
        num_experts: int = 3,
    ):
        super().__init__(
            summary_dim,
            latent_dim,
            action_type_dim,
            cone_context_dim,
            dropout,
            num_experts,
        )
        self.horizon_feature_dim = 4
        expert_dim = max(8, latent_dim // 2)
        self.marginal_rank_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim + self.horizon_feature_dim),
                    nn.Linear(latent_dim + self.horizon_feature_dim, expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_dim, 1),
                )
                for _ in range(num_experts)
            ]
        )
        for expert in self.marginal_rank_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(
        self,
        summary: torch.Tensor,
        action_node_z: torch.Tensor,
        action_type_z: torch.Tensor,
        cone_context: torch.Tensor,
        sequence_step: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        hidden = self._hidden(
            torch.cat([summary, cone_context], dim=0),
            action_node_z,
            action_type_z,
        )
        outputs = self._outputs(hidden)
        gate = torch.softmax(self.expert_gate(action_type_z), dim=0)
        type_values = torch.stack([expert(hidden) for expert in self.type_experts], dim=0)
        type_residual = (gate.view(-1, 1) * type_values).sum(dim=0)
        horizon = TypedConeHorizonMoEUtilityHead._horizon_features(
            sequence_step,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        marginal_input = torch.cat([hidden, horizon], dim=0)
        marginal_values = torch.stack(
            [expert(marginal_input).reshape(()) for expert in self.marginal_rank_experts],
            dim=0,
        )
        marginal_residual = (gate * marginal_values).sum()
        return {
            "typed_marginal_pred": outputs["typed_marginal_pred"] + type_residual[0] + marginal_residual,
            "typed_return_pred": outputs["typed_return_pred"] + type_residual[1],
            "typed_sa_reduction_pred": outputs["typed_sa_reduction_pred"] + type_residual[2:4],
        }


class TPIWorldModel(nn.Module):
    """JEPA-style world model with EMA target encoder and action heads."""

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 64,
        encoder_layers: int = 3,
        action_type_dim: int = 16,
        dropout: float = 0.1,
        head_context: bool = False,
        relation_dim: int = 4,
        edge_weight_mode: str = "mean",
        edge_keep_ratio: float = 1.0,
        residual_dynamics: bool = False,
        relation_gate: bool = False,
        hard_head_type: str = "mlp",
        encoder_type: str = "mean",
        summary_mode: str = "global",
        q_head_type: str = "summary",
        utility_head_type: str = "legacy",
    ):
        super().__init__()
        self.head_context = bool(head_context)
        self.relation_dim = int(relation_dim)
        self.residual_dynamics = bool(residual_dynamics)
        self.relation_gate = bool(relation_gate)
        self.hard_head_type = str(hard_head_type or "mlp").lower()
        self.encoder_type = str(encoder_type or "mean").lower()
        self.summary_mode = str(summary_mode or "global").lower()
        self.q_head_type = str(q_head_type or "summary").lower()
        self.utility_head_type = str(utility_head_type or "legacy").lower()
        self.online_encoder = NodeEncoder(
            feature_dim,
            latent_dim,
            encoder_layers,
            dropout,
            edge_weight_mode=edge_weight_mode,
            edge_keep_ratio=edge_keep_ratio,
            encoder_type=self.encoder_type,
        )
        self.target_encoder = NodeEncoder(
            feature_dim,
            latent_dim,
            encoder_layers,
            dropout,
            edge_weight_mode=edge_weight_mode,
            edge_keep_ratio=edge_keep_ratio,
            encoder_type=self.encoder_type,
        )
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.action_encoder = ActionEncoder(latent_dim, action_type_dim)
        self.dynamics = DynamicsPredictor(latent_dim, self.relation_dim, dropout)
        summary_dim = latent_dim * 2 + 6
        if self.summary_mode in {"cone", "cone_pool", "hybrid"}:
            summary_dim += latent_dim * 3
        if self.head_context:
            summary_dim += latent_dim + self.relation_dim
        self.q_head = nn.Sequential(nn.Linear(summary_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))
        self.q_node_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))
        self.q_type_head = nn.Sequential(
            nn.Linear(summary_dim + action_type_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        self.reward_head = nn.Sequential(nn.Linear(summary_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))
        self.pattern_head = nn.Sequential(nn.Linear(summary_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))
        self.return_head = nn.Sequential(nn.Linear(summary_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))
        self.hard_reduction_head = nn.Sequential(
            nn.Linear(summary_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3),
        )
        self.typed_utility_head: TypedUtilityHead | None = None
        self.typed_cone_utility_head: TypedConeUtilityHead | None = None
        if self.utility_head_type in {"typed", "typed_film", "action_typed"}:
            self.typed_utility_head = TypedUtilityHead(
                summary_dim,
                latent_dim,
                action_type_dim,
                dropout,
            )
        elif self.utility_head_type in {
            "typed_cone",
            "typed_cone_film",
            "action_cone",
            "typed_cone_moe",
            "typed_cone_experts",
            "typed_cone_return_rank_moe",
            "typed_cone_dual_rank_moe",
            "typed_cone_return_horizon_rank_moe",
            "typed_cone_horizon_return_rank_moe",
            "typed_cone_return_late_horizon_rank_moe",
            "typed_cone_late_horizon_return_rank_moe",
            "typed_cone_return_late_type_rank_moe",
            "typed_cone_late_type_return_rank_moe",
            "typed_cone_return_late_control_rank_moe",
            "typed_cone_late_control_return_rank_moe",
            "typed_cone_horizon_moe",
            "typed_cone_horizon_experts",
            "typed_cone_marginal_horizon_rank_moe",
            "typed_cone_marginal_rank_moe",
        }:
            # Three relation pools (fanin, fanout, union), each represented by
            # its current latent and predicted action effect, plus aggregate
            # relation/topology features.
            cone_context_dim = latent_dim * 6 + self.relation_dim
            if self.utility_head_type in {
                "typed_cone_return_late_control_rank_moe",
                "typed_cone_late_control_return_rank_moe",
            }:
                head_cls = TypedConeReturnLateControlRankMoEUtilityHead
            elif self.utility_head_type in {
                "typed_cone_return_late_type_rank_moe",
                "typed_cone_late_type_return_rank_moe",
            }:
                head_cls = TypedConeReturnLateTypeRankMoEUtilityHead
            elif self.utility_head_type in {
                "typed_cone_return_late_horizon_rank_moe",
                "typed_cone_late_horizon_return_rank_moe",
            }:
                head_cls = TypedConeReturnLateHorizonRankMoEUtilityHead
            elif self.utility_head_type in {
                "typed_cone_return_horizon_rank_moe",
                "typed_cone_horizon_return_rank_moe",
            }:
                head_cls = TypedConeReturnHorizonRankMoEUtilityHead
            elif self.utility_head_type in {
                "typed_cone_horizon_moe",
                "typed_cone_horizon_experts",
            }:
                head_cls = TypedConeHorizonMoEUtilityHead
            elif self.utility_head_type in {
                "typed_cone_marginal_horizon_rank_moe",
                "typed_cone_marginal_rank_moe",
            }:
                head_cls = TypedConeMarginalHorizonRankMoEUtilityHead
            elif self.utility_head_type in {
                "typed_cone_return_rank_moe",
                "typed_cone_dual_rank_moe",
            }:
                head_cls = TypedConeReturnRankMoEUtilityHead
            elif self.utility_head_type in {"typed_cone_moe", "typed_cone_experts"}:
                head_cls = TypedConeMoEUtilityHead
            else:
                head_cls = TypedConeUtilityHead
            self.typed_cone_utility_head = head_cls(
                summary_dim,
                latent_dim,
                action_type_dim,
                cone_context_dim,
                dropout,
            )
        elif self.utility_head_type in {"legacy", "none", "off"}:
            pass
        else:
            raise ValueError(f"unsupported utility_head_type={utility_head_type!r}")
        self.scoap_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 3))
        self.delta_scoap_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 3))
        if self.hard_head_type in {"residual", "residual_context", "context"}:
            use_relation = self.hard_head_type in {"residual_context", "context"}
            self.hard_head = ResidualHardHead(latent_dim, self.relation_dim, 2, dropout, use_relation=use_relation)
            self.hard_count_head = ResidualHardHead(latent_dim, self.relation_dim, 1, dropout, use_relation=use_relation)
        else:
            self.hard_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 2))
            self.hard_count_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))

    def _summary(
        self,
        z: torch.Tensor,
        action_node_id: int | None = None,
        relation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress all predicted node latents into one graph/action summary vector."""

        norms = z.norm(dim=1)
        k = min(20, z.shape[0])
        top_idx = torch.topk(norms, k=k).indices
        extras = torch.stack(
            [
                norms.mean(),
                norms.max(),
                norms.min(),
                norms.std(unbiased=False),
                norms.median(),
                torch.tensor(float(z.shape[0]), dtype=z.dtype, device=z.device).log1p(),
            ]
        )
        parts = [z.mean(dim=0), z[top_idx].mean(dim=0), extras]
        if self.summary_mode in {"cone", "cone_pool", "hybrid"}:
            if relation_features is None:
                raise ValueError("cone summary requires relation_features")
            rel = relation_features.to(device=z.device, dtype=z.dtype)

            def weighted_mean(weights: torch.Tensor) -> torch.Tensor:
                weights = weights.view(-1, 1).clamp_min(0.0)
                denom = weights.sum().clamp_min(1.0)
                return (z * weights).sum(dim=0) / denom

            if rel.shape[1] > 6:
                parts.append(weighted_mean(rel[:, 6]))
            else:
                parts.append(weighted_mean(rel[:, 1] + rel[:, 2]))
            parts.append(weighted_mean(rel[:, 4] if rel.shape[1] > 4 else rel[:, 1]))
            parts.append(weighted_mean(rel[:, 5] if rel.shape[1] > 5 else rel[:, 2]))
        if self.head_context:
            if action_node_id is None or relation_features is None:
                raise ValueError("head_context requires action_node_id and relation_features")
            parts.extend([z[action_node_id], relation_features.mean(dim=0)])
        return torch.cat(parts, dim=0)

    def _typed_cone_context(
        self,
        z_t: torch.Tensor,
        z_pred: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> torch.Tensor:
        """Pool current state and predicted effect over action-centric cones."""

        rel = relation_features.to(device=z_t.device, dtype=z_t.dtype)
        if rel.shape[1] > 6:
            fanin_weight = rel[:, 4]
            fanout_weight = rel[:, 5]
            cone_weight = rel[:, 6]
        else:
            fanin_weight = rel[:, 1]
            fanout_weight = rel[:, 2]
            cone_weight = (rel[:, 0] + rel[:, 1] + rel[:, 2]).clamp(max=1.0)

        def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            weights = weights.view(-1, 1).clamp_min(0.0)
            return (values * weights).sum(dim=0) / weights.sum().clamp_min(1.0)

        delta = z_pred - z_t
        parts = []
        for weights in (fanin_weight, fanout_weight, cone_weight):
            parts.append(weighted_mean(z_t, weights))
            parts.append(weighted_mean(delta, weights))
        parts.append(rel.mean(dim=0))
        return torch.cat(parts, dim=0)

    @staticmethod
    def _derived_hard_counts(hard_logits: torch.Tensor) -> torch.Tensor:
        """Return differentiable graph-level hard counts from node hard probabilities."""

        prob = hard_logits.sigmoid()
        sa0 = prob[:, 0].sum()
        sa1 = prob[:, 1].sum()
        total = sa0 + sa1
        return torch.stack([total, sa0, sa1])

    @staticmethod
    def _derived_hard_reduction(pre_counts: torch.Tensor, post_counts: torch.Tensor) -> torch.Tensor:
        """Return normalized hard-count reduction ratios from pre/post counts."""

        return ((pre_counts - post_counts) / pre_counts.clamp_min(1.0)).clamp(-1.0, 1.0)

    def predict_from_latent(
        self,
        z_t: torch.Tensor,
        action_node_id: int,
        action_type_id: int | torch.Tensor,
        relation_features: torch.Tensor,
        include_aux_heads: bool = True,
        sequence_step: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        """Predict the next latent state from an already-encoded current state."""

        action_emb = self.action_encoder(z_t[action_node_id], action_type_id)
        z_update = self.dynamics(z_t, action_emb, relation_features)
        if self.residual_dynamics:
            if self.relation_gate:
                gate = relation_features[:, 3:4].clamp(0.0, 1.0)
                if relation_features.shape[1] > 6:
                    gate = torch.maximum(gate, relation_features[:, 6:7].clamp(0.0, 1.0))
                gate = 0.10 + 0.90 * gate
                z_pred = z_t + gate * z_update
            else:
                z_pred = z_t + z_update
        else:
            z_pred = z_update
        summary = self._summary(z_pred, action_node_id, relation_features)
        if self.q_head_type in {"factorized", "node_type", "node_type_factorized"}:
            if not torch.is_tensor(action_type_id):
                type_id = torch.tensor(action_type_id, dtype=torch.long, device=z_t.device)
            else:
                type_id = action_type_id.to(device=z_t.device, dtype=torch.long).view(())
            type_z = self.action_encoder.type_emb(type_id)
            q_node = self.q_node_head(z_t[action_node_id]).view(())
            q_type = self.q_type_head(torch.cat([summary, type_z], dim=0)).view(())
            q_pred = q_node + q_type
        else:
            q_pred = self.q_head(summary).view(())
        reward_pred = self.reward_head(summary).view(())
        pre_hard_logits = self._hard_logits(z_t, relation_features)
        post_hard_logits = self._hard_logits(z_pred, relation_features)
        derived_pre_counts = self._derived_hard_counts(pre_hard_logits)
        derived_post_counts = self._derived_hard_counts(post_hard_logits)
        out = {
            "z_pred": z_pred,
            "q_pred": q_pred,
            "reward_pred": reward_pred,
            "fc_pred": reward_pred,
            "pattern_pred": self.pattern_head(summary).view(()),
            "score_pred": q_pred,
            "return_pred": self.return_head(summary).view(()),
            "hard_reduction_pred": self.hard_reduction_head(summary),
            "pre_hard_logits": pre_hard_logits,
            "derived_hard_count_pre_pred": derived_pre_counts,
            "derived_hard_count_post_pred": derived_post_counts,
            "derived_hard_reduction_pred": self._derived_hard_reduction(derived_pre_counts, derived_post_counts),
        }
        if self.typed_utility_head is not None:
            if not torch.is_tensor(action_type_id):
                typed_action_id = torch.tensor(action_type_id, dtype=torch.long, device=z_t.device)
            else:
                typed_action_id = action_type_id.to(device=z_t.device, dtype=torch.long).view(())
            out.update(
                self.typed_utility_head(
                    summary,
                    z_t[action_node_id],
                    self.action_encoder.type_emb(typed_action_id),
                )
            )
        elif self.typed_cone_utility_head is not None:
            if not torch.is_tensor(action_type_id):
                typed_action_id = torch.tensor(action_type_id, dtype=torch.long, device=z_t.device)
            else:
                typed_action_id = action_type_id.to(device=z_t.device, dtype=torch.long).view(())
            typed_args = (
                summary,
                z_t[action_node_id],
                self.action_encoder.type_emb(typed_action_id),
                self._typed_cone_context(z_t, z_pred, relation_features),
            )
            if isinstance(
                self.typed_cone_utility_head,
                TypedConeReturnLateControlRankMoEUtilityHead,
            ):
                out.update(
                    self.typed_cone_utility_head(
                        *typed_args,
                        sequence_step=sequence_step,
                        action_type_id=typed_action_id,
                    )
                )
            elif isinstance(
                self.typed_cone_utility_head,
                (
                    TypedConeHorizonMoEUtilityHead,
                    TypedConeReturnHorizonRankMoEUtilityHead,
                    TypedConeReturnLateHorizonRankMoEUtilityHead,
                    TypedConeReturnLateTypeRankMoEUtilityHead,
                    TypedConeMarginalHorizonRankMoEUtilityHead,
                ),
            ):
                out.update(self.typed_cone_utility_head(*typed_args, sequence_step=sequence_step))
            else:
                out.update(self.typed_cone_utility_head(*typed_args))
        if include_aux_heads:
            out.update(
                {
                    "scoap_pred": self.scoap_head(z_pred),
                    "delta_scoap_pred": self.delta_scoap_head(z_pred),
                    "hard_logits": post_hard_logits,
                    "hard_count_pred": self._hard_count(z_pred, relation_features),
                }
            )
        return out

    def _hard_logits(self, z: torch.Tensor, relation_features: torch.Tensor) -> torch.Tensor:
        if isinstance(self.hard_head, ResidualHardHead):
            return self.hard_head(z, relation_features)
        return self.hard_head(z)

    def _hard_count(self, z: torch.Tensor, relation_features: torch.Tensor) -> torch.Tensor:
        if isinstance(self.hard_count_head, ResidualHardHead):
            return self.hard_count_head(z, relation_features).squeeze(-1)
        return self.hard_count_head(z).squeeze(-1)

    def forward(
        self,
        graph,
        x_pre: torch.Tensor,
        x_post: torch.Tensor,
        action_node_id: int,
        action_type_id: int | torch.Tensor,
        relation_features: torch.Tensor,
        sequence_step: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        """Run one transition through the model."""

        edge_src = graph.edge_src.to(x_pre.device)
        edge_dst = graph.edge_dst.to(x_pre.device)
        gate_type_ids = graph.gate_type_ids.to(x_pre.device)
        z_t = self.online_encoder(x_pre, edge_src, edge_dst, gate_type_ids)
        with torch.no_grad():
            target_was_training = self.target_encoder.training
            self.target_encoder.eval()
            z_t1 = self.target_encoder(x_post, edge_src, edge_dst, gate_type_ids)
            self.target_encoder.train(target_was_training)
        pred = self.predict_from_latent(
            z_t,
            action_node_id,
            action_type_id,
            relation_features,
            sequence_step=sequence_step,
        )
        return {
            "z_t": z_t,
            "z_t1": z_t1,
            **pred,
        }


@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, decay: float) -> None:
    """Update target parameters toward online parameters by EMA."""

    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)


def _main() -> None:
    """CLI summary used by the step-by-step validation plan."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    args = parser.parse_args()

    all_rows = load_labels(Path(args.labels))
    rows = [row for row in all_rows if row.benchmark_id == "iscas89__s838"]
    if not rows:
        rows = all_rows
    dataset = TPIDataset(rows, max_specs=1)
    sample = dataset[0]
    model = TPIWorldModel(feature_dim=sample.x_pre.shape[1])
    out = model(
        sample.graph,
        sample.x_pre,
        sample.x_post,
        sample.action_node_id,
        sample.action_type_id,
        sample.relation_features,
    )
    print(f"z_pred_shape={tuple(out['z_pred'].shape)}")
    print(f"z_t_shape={tuple(out['z_t'].shape)}")
    print(f"q_scalar={out['q_pred'].ndim == 0}")
    print(f"scoap_pred_shape={tuple(out['scoap_pred'].shape)}")
    print(f"fc_scalar={out['fc_pred'].ndim == 0}")
    print(f"pattern_scalar={out['pattern_pred'].ndim == 0}")
    print(f"score_is_q={bool(torch.equal(out['score_pred'], out['q_pred']))}")
    print(f"return_scalar={out['return_pred'].ndim == 0}")
    print(f"hard_logits_shape={tuple(out['hard_logits'].shape)}")
    print(f"hard_count_shape={tuple(out['hard_count_pred'].shape)}")
    print(f"hard_reduction_shape={tuple(out['hard_reduction_pred'].shape)}")
    print(f"finite={bool(torch.isfinite(out['z_pred']).all().item())}")


if __name__ == "__main__":
    _main()
