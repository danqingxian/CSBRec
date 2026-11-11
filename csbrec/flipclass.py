"""Flipped-classroom training framework of FCRLLM, with CSBRec sub-models.

Adapted from FCRLLM's models/FlipClassroom.py (https://github.com/Byungmoon-Heo/FCRLLM, MIT licence, see
THIRD_PARTY_NOTICES.md): hopfield_update_full, the attention block, FlipBlockST / FlipBlockCT, FCRLLM's
PointWiseFeedForward and the input views. The only change is that the diagonal term of the Hopfield update is written as
a row scaling instead of a (B L) x (B L) diagonal matrix product (same values and gradients, less memory).

Two parameter-independent CSBRec sub-models are trained on the same batches:
  ST  the semantic view is the teacher (the collaborative view is the student)
  CT  the collaborative view is the teacher (the semantic view is the student)
For each sub-model, a one-layer attention block encodes both embedding sequences; the teacher sequence K is updated by
    K~ = K + eta * [Softmax(mu K Q^T) Q - rho K - D K],   D = Diag(Softmax(||k_i||^2 / 2)),
and the alignment loss is the MSE between the student sequence Q and K~.
Total loss: L = 1/2 (L_CSB^ST + lambda_fc L_align^ST) + 1/2 (L_CSB^CT + lambda_fc L_align^CT).
At inference the two sub-models' scores are averaged.
Symbols: mu, rho, eta, lambda_fc are FCRLLM's beta_hopfield, alpha_hopfield, gamma_update, gamma_fc.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCPointWiseFeedForward(nn.Module):
    """FCRLLM models/utils.py PointWiseFeedForward."""

    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units * 4)
        self.act = self.gelu
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_units * 4, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.LayerNorm = nn.LayerNorm(hidden_units, eps=1e-12)

    def forward(self, inputs) -> torch.Tensor:
        x = self.linear1(inputs)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.LayerNorm(x + inputs)
        return x

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def hopfield_update_full(teacherK, studentQ, beta=1.0, alpha=0.1, gamma_update=0.5, diag_reg=True):
    B, L, d = teacherK.shape
    t2d = teacherK.reshape(B * L, d)
    s2d = studentQ.reshape(B * L, d)
    attn = F.softmax(beta * (t2d @ s2d.T), dim=1)
    align = attn @ s2d
    reg = alpha * t2d
    if diag_reg:
        row_sm = torch.softmax(0.5 * torch.sum(t2d * t2d, dim=1), dim=0)
        diag = row_sm[:, None] * t2d          # FCRLLM: torch.diag(row_sm) @ t2d
    else:
        diag = torch.zeros_like(t2d)
    updated = t2d + gamma_update * (align - reg - diag)
    return updated.reshape(B, L, d)


class _HopBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads)
        self.ffn = FCPointWiseFeedForward(hidden, dropout)

    def forward(self, x: torch.Tensor):
        x = self.ln(x.transpose(0, 1))
        attn_out, _ = self.attn(x, x, x)
        out = x + attn_out
        return self.ffn(out.transpose(0, 1))


class FlipBlockST(nn.Module):
    """Semantic view as teacher."""

    def __init__(self, hidden, heads, dropout, *, beta, alpha, gamma, num_hopblock=1):
        super().__init__()
        self.student_layers = nn.ModuleList([_HopBlock(hidden, heads, dropout) for _ in range(num_hopblock)])
        self.teacher_layers = nn.ModuleList([_HopBlock(hidden, heads, dropout) for _ in range(num_hopblock)])
        self.cfg = dict(beta=beta, alpha=alpha, gamma_update=gamma)

    def forward(self, collab, semantic):
        s, t = collab, semantic
        for layer_s, layer_t in zip(self.student_layers, self.teacher_layers):
            s = layer_s(s)
            t = layer_t(t)
        return s, hopfield_update_full(t, s, **self.cfg)


class FlipBlockCT(nn.Module):
    """Collaborative view as teacher."""

    def __init__(self, hidden, heads, dropout, *, beta, alpha, gamma, num_hopblock=1):
        super().__init__()
        self.teacher_layers = nn.ModuleList([_HopBlock(hidden, heads, dropout) for _ in range(num_hopblock)])
        self.student_layers = nn.ModuleList([_HopBlock(hidden, heads, dropout) for _ in range(num_hopblock)])
        self.cfg = dict(beta=beta, alpha=alpha, gamma_update=gamma)

    def forward(self, collab, semantic):
        t, s = collab, semantic
        for layer_t, layer_s in zip(self.teacher_layers, self.student_layers):
            t = layer_t(t)
            s = layer_s(s)
        return hopfield_update_full(t, s, **self.cfg), s


class CSBRec(nn.Module):
    """Two CSBRec sub-models (ST, CT) in the flipped-classroom framework."""

    def __init__(self, Model, user_num, item_num, device, args):
        super().__init__()
        self.hyb, self.hid, self.lambda_fc = 0.5, args.hidden_size, args.lambda_fc
        blk = dict(beta=args.mu, alpha=args.rho, gamma=args.eta, num_hopblock=1)
        # construction order as in FCRLLM's FlipClassEnsemble: st's encoder, its block, ct's encoder, its block
        self.st = Model(user_num, item_num, device, args)
        self.blk_st = FlipBlockST(self.hid, args.num_heads, args.dropout_rate, **blk)
        self.ct = Model(user_num, item_num, device, args)
        self.blk_ct = FlipBlockCT(self.hid, args.num_heads, args.dropout_rate, **blk)
        self.mse = nn.MSELoss()
        self.d = 2 * self.hid   # width of each view in user_reps / item_tables
        self.parts = {}

    @property
    def subs(self):
        return [self.st, self.ct]

    def _views(self, m, seq, positions):
        """FCRLLM _BaseFlip._views on sub-model m's tables."""
        pos = positions.long()
        co = m.id_item_emb(seq) * (self.hid ** 0.5)
        co = m.emb_dropout(co + m.pos_emb(pos))
        se = m.adapter(m.llm_item_emb(seq)) * (self.hid ** 0.5)
        se = m.emb_dropout(se + m.pos_emb(pos))
        return co, se

    def _align(self, m, ct_dir, seq, positions):
        co, se = self._views(m, seq, positions)
        if ct_dir:
            teach_upd, stud = self.blk_ct(co, se)
            return self.mse(teach_upd, stud)
        stud, teach_upd = self.blk_st(co, se)
        return self.mse(stud, teach_upd)

    def sub_loss(self, k, seq, pos, neg, positions, **kw):
        m = self.subs[k]
        loss = m(seq, pos, neg, positions, **kw)
        parts = dict(m.parts)
        if self.lambda_fc > 0:
            a = self._align(m, k == 1, seq, positions)
            loss = loss + self.lambda_fc * a
            parts["fc_align"] = a.item()
        return loss, parts

    def forward(self, seq, pos, neg, positions, **kw):
        """Total loss in one graph (used by the checks; training runs step_backward)."""
        l0, p0 = self.sub_loss(0, seq, pos, neg, positions, **kw)
        l1, p1 = self.sub_loss(1, seq, pos, neg, positions, **kw)
        self.parts = {k: 0.5 * (p0.get(k, 0.0) + p1.get(k, 0.0)) for k in set(p0) | set(p1)}
        return self.hyb * l0 + (1 - self.hyb) * l1

    def step_backward(self, inputs):
        """Backward of 1/2 L_ST + 1/2 L_CT, one sub-model at a time (they share no parameter: same gradients, half the
        peak memory). Returns the loss value."""
        total, ps = 0.0, []
        for k, w in ((0, self.hyb), (1, 1.0 - self.hyb)):
            loss, p = self.sub_loss(k, **inputs)
            (w * loss).backward()
            total += w * loss.item()
            ps.append(p)
        self.parts = {k: 0.5 * (ps[0].get(k, 0.0) + ps[1].get(k, 0.0)) for k in set(ps[0]) | set(ps[1])}
        return total

    @torch.no_grad()
    def user_reps(self, seq, positions):
        d, w = self.hid, self.hyb
        h0, h1 = self.st.user_reps(seq, positions), self.ct.user_reps(seq, positions)
        return torch.cat([w * h0[:, :d], (1 - w) * h1[:, :d], w * h0[:, d:], (1 - w) * h1[:, d:]], dim=-1)

    @torch.no_grad()
    def item_tables(self):
        c0, s0 = self.st.item_tables()
        c1, s1 = self.ct.item_tables()
        return torch.cat([c0, c1], dim=-1), torch.cat([s0, s1], dim=-1)

    def set_ngg_beta(self, beta):
        for m in self.subs:
            m.set_ngg_beta(beta)
