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
        pred = self.predict_from_latent(z_t, action_node_id, action_type_id, relation_features)
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
