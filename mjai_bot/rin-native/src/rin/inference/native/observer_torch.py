"""FP32 public Actor-token readout, equivalent to the versioned Flax head."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rin.actions import ActionKind
from .observer_checkpoint import ARCHITECTURE, KIND, PAYMENT_BUCKET_EDGES


class ObserverReadoutTorch(nn.Module):
    def __init__(self, arrays, config, *, device="cpu", attention="sdpa"):
        super().__init__()
        self.config = config["observer"]
        self.attention = attention
        self.weights = nn.ParameterDict()
        for name, array in arrays.items():
            value = torch.from_numpy(array.copy())
            if name.endswith("/kernel"):
                if value.ndim == 3:
                    value = value.reshape(-1, value.shape[-1]) if "/out/" in name else value.reshape(value.shape[0], -1)
                value = value.T.contiguous()
            elif name.endswith("/bias") and value.ndim == 2:
                value = value.flatten()
            self.weights[name.replace("/", "__")] = nn.Parameter(value.to(device=device, dtype=torch.float32), requires_grad=False)
        self.register_buffer("joint_bits", ((torch.arange(8, device=device)[:, None] >> torch.arange(3, device=device)[None]) & 1).float())
        self.eval()

    def w(self, name):
        return self.weights[name.replace("/", "__")]

    def dense(self, value, name):
        return F.linear(value.float(), self.w(name + "/kernel"), self.w(name + "/bias"))

    def norm(self, value, name):
        value = value.float()
        return value * torch.rsqrt((value * value).mean(-1, keepdim=True) + 1e-6) * self.w(name + "/scale")

    @torch.inference_mode()
    def forward(self, representations, candidates, candidate_mask):
        cfg = self.config
        if set(representations) != {"state", "tile_tokens", "history_tokens", "history_mask", "global_state"}:
            raise ValueError("Observer requires only the versioned public Actor representations")
        visible = {name: value.detach() for name, value in representations.items()}
        batch = visible["state"].shape[0]
        history_mask = visible["history_mask"].bool()
        width, heads = cfg["width"], cfg["heads"]
        expected = {"state": (batch, cfg["state_width"]), "tile_tokens": (batch, 34, cfg["tile_width"]),
                    "history_tokens": (*history_mask.shape, cfg["history_width"]), "global_state": (batch, cfg["global_width"])}
        if history_mask.ndim != 2 or any(tuple(visible[name].shape) != shape for name, shape in expected.items()):
            raise ValueError("Observer representation shapes differ from the bound Actor")
        def project(name, value, index):
            return self.dense(value, name + "_projection") + self.w("memory_type_embedding")[index]
        memory = torch.cat((project("state", visible["state"][:, None], 0),
                            project("tile", visible["tile_tokens"], 1),
                            project("history", torch.where(history_mask[..., None], visible["history_tokens"], 0), 2),
                            project("global", visible["global_state"][:, None], 3)), dim=1)
        ones = lambda length: torch.ones((batch, length), dtype=torch.bool, device=memory.device)
        memory_mask = torch.cat((ones(35), history_mask, ones(1)), dim=1)
        opponent_query = self.w("opponent_query")
        wait_query = (opponent_query[:, None] + self.w("wait_tile_query")[None]).reshape(102, width)
        def embed(name, field):
            table = self.w("candidate_" + name + "/embedding")
            return F.embedding((candidates[field].long() + 1).clamp(0, len(table) - 1), table)
        candidate_query = sum(embed(name, field) for name, field in
                              (("tile", "tile"), ("kind", "kind"), ("from_draw", "from_draw"),
                               ("red_mask", "red_tile_mask"), ("phase", "decision_phase")))
        query = torch.cat((opponent_query[None].expand(batch, -1, -1),
                           wait_query[None].expand(batch, -1, -1), candidate_query), dim=1)
        applicable = candidate_mask.bool() & ((candidates["kind"] == int(ActionKind.DISCARD)) |
                                               (candidates["kind"] == int(ActionKind.RIICHI_DISCARD)))
        query_mask = torch.cat((ones(105), applicable), dim=1)
        mask = query_mask[:, None, :, None] & memory_mask[:, None, None, :]
        normalized_query, normalized_memory = self.norm(query, "query_norm"), self.norm(memory, "memory_norm")
        q = self.dense(normalized_query, "readout_attention/query").reshape(batch, -1, heads, width // heads).transpose(1, 2)
        k, v = [self.dense(normalized_memory, "readout_attention/" + name).reshape(batch, -1, heads, width // heads).transpose(1, 2)
                for name in ("key", "value")]
        q = q / math.sqrt(width // heads)
        if self.attention == "sdpa":
            bias = torch.where(mask, 0.0, torch.finfo(q.dtype).min)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0., scale=1.)
        else:
            scores = (q @ k.transpose(-1, -2)).masked_fill(~mask, torch.finfo(q.dtype).min)
            attended = scores.softmax(-1) @ v
        attended = attended.transpose(1, 2).reshape(batch, -1, width)
        hidden = query + self.dense(attended, "readout_attention/out")
        gate, value = self.dense(self.norm(hidden, "mlp_norm"), "mlp_in").chunk(2, dim=-1)
        hidden = self.norm(hidden + self.dense(F.silu(gate) * value, "mlp_out"), "output_norm")
        candidate_hidden = hidden[:, 105:]
        tile = candidates["tile"].long()
        red_base = torch.tensor([4, 13, 22], device=tile.device)
        tile_type = torch.where(tile >= 34, red_base[(tile - 34).clamp(0, 2)], tile)
        output = {
            "tenpai_logits": self.dense(hidden[:, :3], "tenpai_head").squeeze(-1),
            "conditional_wait_logits": self.dense(hidden[:, 3:105], "conditional_wait_head").reshape(batch, 3, 34),
            "ron_joint_logits": torch.where(applicable[..., None], self.dense(candidate_hidden, "ron_joint_head"), 0.),
            "conditional_loss_points": torch.where(applicable, F.softplus(self.dense(candidate_hidden, "conditional_loss_head").squeeze(-1)) * cfg["loss_point_scale"], 0.),
            "candidate_applicable": applicable,
            "candidate_tile_type": torch.where(applicable, tile_type, -1),
        }
        if cfg["payment_distribution"]:
            output["conditional_payment_logits"] = torch.where(applicable[..., None], self.dense(candidate_hidden, "conditional_payment_distribution_head"), 0.)
        return output

    def probabilities(self, output):
        tenpai = output["tenpai_logits"].sigmoid()
        waits = output["conditional_wait_logits"].sigmoid()
        unconditional = tenpai[..., None] * waits
        applicable = output["candidate_applicable"]
        joint = torch.where(applicable[..., None], output["ron_joint_logits"].softmax(-1), 0.)
        ron = joint @ self.joint_bits
        any_ron = torch.where(applicable, 1. - joint[..., 0], 0.)
        tiles = output["candidate_tile_type"].clamp(0, 33)
        candidate_wait = torch.gather(unconditional, 2, tiles[:, None].expand(-1, 3, -1)).transpose(1, 2)
        result = {"tenpai_probability": tenpai, "conditional_wait_probability": waits,
                  "unconditional_wait_probability": unconditional, "ron_joint_probability": joint,
                  "ron_probability": ron, "any_ron_probability": any_ron,
                  "multiple_ron_probability": (joint * (self.joint_bits.sum(-1) >= 2)).sum(-1),
                  "conditional_loss_points": output["conditional_loss_points"],
                  "expected_loss_points": any_ron * output["conditional_loss_points"],
                  "ron_wait_inclusion_violation": torch.where(applicable[..., None], (ron - candidate_wait).clamp_min(0), 0.),
                  "candidate_applicable": applicable}
        if "conditional_payment_logits" in output:
            distribution = torch.where(applicable[..., None], output["conditional_payment_logits"].softmax(-1), 0.)
            result["conditional_payment_probability"] = distribution
            for threshold in (8000, 12000, 24000):
                conditional = distribution[..., sum(edge <= threshold for edge in PAYMENT_BUCKET_EDGES):].sum(-1)
                result[f"conditional_payment_ge_{threshold}"] = conditional
                result[f"payment_ge_{threshold}"] = any_ron * conditional
        return result


def observer_metadata(predictor, *, seat, row=0, continuation=False):
    identity = getattr(predictor, "observer_manifest", None)
    observer_identity = None if identity is None else {
        "source_sha256": identity["source_sha256"], "safetensors_sha256": identity["observer_sha256"],
        "config_sha256": identity["config_sha256"], "representation_actor_sha256": identity["representation_actor_sha256"]}
    base = {"actor_identity": predictor.manifest["actor_sha256"], "observer_identity": observer_identity}
    status = getattr(predictor, "observer_status", "unavailable")
    if status != "ready":
        return {**base, "status": status, "reason": getattr(predictor, "observer_reason", "no_compatible_observer")}
    if continuation:
        return {**base, "status": "not_evaluated", "reason": "riichi_continuation"}
    output = predictor.last_observer_outputs
    if output is None:
        return {**base, "status": "error", "reason": "observer_output_missing"}
    order = [(seat + relative) % 4 for relative in (1, 2, 3)]
    opponents = [{"seat": actor, "relative_seat": index + 1,
                  "tenpai_probability": float(output["tenpai_probability"][row, index]),
                  "conditional_wait_probability": output["conditional_wait_probability"][row, index].tolist(),
                  "unconditional_wait_probability": output["unconditional_wait_probability"][row, index].tolist()}
                 for index, actor in enumerate(order)]
    excluded = {"tenpai_probability", "conditional_wait_probability", "unconditional_wait_probability", "candidate_applicable"}
    candidates = [{"candidate_index": int(index), **{name: value[row, index].tolist() for name, value in output.items() if name not in excluded}}
                  for index in np.flatnonzero(output["candidate_applicable"][row])]
    result = {**base, "status": "ready", "schema_version": ARCHITECTURE, "kind": KIND,
              "opponent_order": order, "opponents": opponents, "candidates": candidates}
    if "conditional_payment_probability" in output:
        result["payment_bucket_edges"] = list(PAYMENT_BUCKET_EDGES)
    return result
