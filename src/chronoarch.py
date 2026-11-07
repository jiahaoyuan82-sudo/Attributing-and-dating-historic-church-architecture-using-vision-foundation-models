"""ChronoArch-F and ChronoArch-L architectures from the v2 experiment plan."""
import math

import torch
from torch import nn
from torch.nn import functional as F


YEAR_EDGES = torch.arange(800, 2051, 50, dtype=torch.float32)
YEAR_CENTERS = (YEAR_EDGES[:-1] + YEAR_EDGES[1:]) / 2


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=384, heads=6, dropout=0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, queries, tokens, return_attention=False):
        q = self.q_norm(queries)
        t = self.token_norm(tokens)
        update, weights = self.attn(q, t, t, need_weights=return_attention)
        queries = queries + update
        queries = queries + self.ff(self.ff_norm(queries))
        return queries, weights


class ChronoArchF(nn.Module):
    def __init__(self, style_classes, fine_classes, *, queries=4, dropout=0.1,
                 token_dim=1024, clip_dim=768, sig_dim=768, tau=0.0,
                 class_counts=None, task="style", variant="A0"):
        super().__init__()
        assert queries in (4, 8)
        assert 0 <= dropout < 1
        self.style_classes = style_classes
        self.fine_classes = fine_classes
        self.queries = queries
        self.tau = tau
        assert task in ("style", "authenticity", "dating")
        self.task = task
        self.variant = variant
        assert variant in ("A0", "A1", "A1a", "A1b", "A2", "A2a", "A2b",
                           "A3", "A3a", "A3b", "A4", "R1", "R2", "R3", "R4")
        self.register_buffer("year_centers", YEAR_CENTERS.clone())
        if class_counts is None:
            class_counts = torch.ones(style_classes, dtype=torch.float32)
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        assert counts.shape == (style_classes,) and torch.all(counts > 0)
        self.register_buffer("class_log_prior", counts.log() - counts.sum().log())

        self.scale_emb = nn.Parameter(torch.zeros(2, token_dim))
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_proj = nn.Linear(token_dim, 384)
        self.query = nn.Parameter(torch.randn(queries, 384) / math.sqrt(384))
        self.cross = nn.ModuleList([CrossAttentionBlock(dropout=dropout) for _ in range(2)])
        self.pool = nn.Linear(queries * 384, 384)
        self.cls_skip = nn.Linear(2 * token_dim, 384)
        if variant in ("A1", "A1a", "A1b", "A4"):
            self.cls_single = nn.Linear(token_dim, 384)

        self.clip_proj = nn.Sequential(nn.LayerNorm(clip_dim), nn.Linear(clip_dim, 384))
        self.sig_proj = nn.Sequential(nn.LayerNorm(sig_dim), nn.Linear(sig_dim, 384))
        self.gate = nn.Linear(3 * 384, 3 * 384)
        if variant == "A2a":
            self.concat_fusion = nn.Linear(3 * 384, 384)
        self.dropout = nn.Dropout(dropout)
        self.style_head = nn.Linear(384, style_classes)
        self.fine_head = nn.Linear(384, fine_classes)
        self.year_head = nn.Linear(384, len(YEAR_CENTERS))

    def forward(self, t224, t448, cls224, cls448, z_clip, z_siglip,
                return_attention=False):
        assert t224.shape[1:] == t448.shape[1:] == (64, self.token_proj.in_features)
        batch = t224.shape[0]
        attention = []
        if self.variant in ("A1", "A4"):
            h_dino = self.cls_single(cls224)
        else:
            if self.variant == "A1a":
                tokens = t224 + self.scale_emb[0]
            elif self.variant == "A1b":
                tokens = t448 + self.scale_emb[1]
            else:
                tokens = torch.cat((t224 + self.scale_emb[0], t448 + self.scale_emb[1]), dim=1)
            tokens = self.token_proj(self.token_norm(tokens))
            query = self.query.unsqueeze(0).expand(batch, -1, -1)
            for block in self.cross:
                query, weights = block(query, tokens, return_attention=return_attention)
                if return_attention:
                    attention.append(weights)
            if self.variant == "A1a":
                h_dino = self.pool(query.flatten(1)) + self.cls_single(cls224)
            elif self.variant == "A1b":
                h_dino = self.pool(query.flatten(1)) + self.cls_single(cls448)
            else:
                h_dino = self.pool(query.flatten(1)) + self.cls_skip(torch.cat((cls224, cls448), dim=-1))
        h_clip = self.clip_proj(z_clip)
        h_sig = self.sig_proj(z_siglip)
        if self.variant in ("A2", "A4"):
            h = h_dino
            gates = None
        elif self.variant == "A2a":
            h = self.concat_fusion(torch.cat((h_dino, h_clip, h_sig), dim=-1))
            gates = None
        else:
            gate_sig = torch.zeros_like(h_sig) if self.variant == "A2b" else h_sig
            gates = self.gate(torch.cat((h_dino, h_clip, gate_sig), dim=-1)).reshape(batch, 3, 384)
            if self.variant == "A2b":
                gates = torch.cat((gates[:, :2].softmax(dim=1),
                                   torch.zeros_like(gates[:, 2:3])), dim=1)
            else:
                gates = gates.softmax(dim=1)
            h = (gates * torch.stack((h_dino, h_clip, h_sig), dim=1)).sum(dim=1)
        h = self.dropout(h)
        style_logits = self.style_head(h)
        fine_logits = self.fine_head(h)
        year_logits = self.year_head(h)
        output = {"style_logits": style_logits, "fine_logits": fine_logits,
                  "year_logits": year_logits, "year_prediction":
                  (year_logits.softmax(dim=-1) * self.year_centers).sum(dim=-1),
                  "gates": gates}
        if return_attention:
            output["attention"] = attention
        return output

    def loss(self, output, style_target, fine_target=None, year_target=None,
             lambda_fine=0.3, lambda_year=0.3):
        style_logits = output["style_logits"] + self.tau * self.class_log_prior
        style_loss = F.cross_entropy(style_logits, style_target)
        fine_loss = style_loss.new_zeros(())
        if fine_target is not None:
            valid = fine_target >= 0
            if valid.any():
                fine_loss = F.cross_entropy(output["fine_logits"][valid], fine_target[valid])
        year_loss = style_loss.new_zeros(())
        if year_target is not None:
            valid = torch.isfinite(year_target) & (year_target >= 800) & (year_target <= 2050)
            if valid.any():
                bins = (year_target[valid] - 800) / 50 - 0.5
                centers = torch.arange(len(YEAR_CENTERS), device=year_target.device)
                target = torch.exp(-0.5 * (centers[None, :] - bins[:, None]).square())
                target = target / target.sum(dim=-1, keepdim=True)
                year_loss = F.kl_div(
                    F.log_softmax(output["year_logits"][valid], dim=-1), target,
                    reduction="batchmean")
        if self.variant in ("A3", "A4"):
            lambda_fine = lambda_year = 0
        elif self.variant == "A3a":
            lambda_fine = 0
        elif self.variant == "A3b":
            lambda_year = 0
        if self.task == "dating":
            assert year_target is not None and torch.isfinite(year_target).any()
            return year_loss + lambda_fine * fine_loss + lambda_year * style_loss
        return style_loss + lambda_fine * fine_loss + lambda_year * year_loss

    def readout(self, output):
        if self.task == "dating":
            return output["year_prediction"]
        probabilities = output["style_logits"].softmax(dim=-1)
        return probabilities[:, 1] if self.task == "authenticity" else probabilities


class LoRALinear(nn.Module):
    """Rank-8 residual adapter for a frozen DINOv2 attention projection."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


class ChronoArchL(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        layers = self.backbone.encoder.layer
        assert len(layers) >= 4 and self.backbone.config.hidden_size == head.token_proj.in_features
        for layer in layers[-4:]:
            attention = layer.attention
            for name in ("q_proj", "k_proj", "v_proj"):
                setattr(attention, name, LoRALinear(getattr(attention, name), rank=8, alpha=16))
        self.backbone.gradient_checkpointing_enable()

    def forward(self, pixel224, t448, cls448, z_clip, z_siglip, return_attention=False):
        output = self.backbone(pixel_values=pixel224).last_hidden_state
        assert output.shape[1] == 257
        return self.head(output[:, 1:], t448, output[:, 0], cls448, z_clip, z_siglip,
                         return_attention=return_attention)


def trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
