"""CSBRec sub-model: an LLM-ESR dual-view model (models/LLMESR.py) trained with the three CSBRec modules.

Scores are LLM-ESR's own: s_co(u, v) = h_co(u) . e_co(v) (collaborative view), s_se(u, v) = h_se(u) . Adapter(l_v)
(semantic view), s = s_co + s_se. The three modules act on the training loss only; inference is unchanged.

  NGG   negative-gradient gating. For a negative item v, the gradient that reaches its own collaborative embedding
        e_co(v) is scaled by beta_v (the score value is unchanged). Default (kappa -> inf): beta_v = 1 if v is a
        positive somewhere in the training data, else 0. With a finite kappa: beta_v = min(1, kappa * P_v / P_mean),
        P_v the number of times v is a training positive. The user state and the semantic view see every negative.
  CCS   cross-view compensating supervision. Each view also gets its own BCE on the same positive / negative pairs,
        weighted by how well the OTHER view alone already separates the pair (no gradient through the weight):
        semantic view weight sigmoid(s_co(pos) - s_co(neg)), collaborative view weight sigmoid(s_se(pos) - s_se(neg)).
        L_cv = L_cv^se + L_cv^co.
  IUSP  item-uniform semantic structure preservation. Each step draws B items uniformly from the item set and matches
        the cosine similarities of their adapter outputs to those of their LLM embeddings:
        L_sp = mean_ij (cos(A_i - mean_B A, A_j - mean_B A) - cos(l_i - mean_V l, l_j - mean_V l))^2,
        where the adapter outputs are centred by the mean over the B sampled items and the frozen LLM embeddings by
        the mean over the whole item set V (computed once).

Per sub-model loss: L = L_rec^beta + alpha * L_SD + lambda_cv * L_cv + lambda_sp * L_sp, where L_rec^beta is the
dual-view BCE with NGG and L_SD is LLM-ESR's retrieval-augmented self-distillation (unchanged).
The training pairs are each backbone's own: SASRec every position of the sequence (seq-to-seq), GRU4Rec the last item
of each training sequence (seq-to-one), BERT4Rec the masked positions.
"""
import torch
import torch.nn.functional as F
from models.LLMESR import LLMESR_SASRec, LLMESR_GRU4Rec, LLMESR_Bert4Rec


class CSBMixin:

    def _csb_init(self, item_num, args):
        self.d = args.hidden_size
        self.n_items = item_num
        self.use_ngg = args.ngg != "none"
        self.lambda_cv, self.lambda_sp, self.sp_items = args.lambda_cv, args.lambda_sp, args.sp_items
        self.register_buffer("ngg_beta", torch.ones(item_num + 2), persistent=False)
        if self.lambda_sp > 0:
            L = self.llm_item_emb.weight.detach()[1:item_num + 1]
            Lc = F.normalize(L - L.mean(0, keepdim=True), dim=-1)
            Lc = torch.cat([torch.zeros_like(Lc[:1]), Lc, torch.zeros_like(Lc[:1])])
            self.register_buffer("llm_c", Lc, persistent=False)
        self.parts = {}

    def set_ngg_beta(self, beta):
        """beta: array over items 1..n."""
        self.ngg_beta[1:self.n_items + 1] = torch.as_tensor(beta, dtype=torch.float32, device=self.ngg_beta.device)

    def sp_loss(self):
        """IUSP: similarity structure of B uniformly drawn items, adapter outputs vs centred LLM embeddings."""
        items = torch.randint(1, self.n_items + 1, (self.sp_items,), device=self.dev)
        A = self.adapter(self.llm_item_emb(items))
        An = F.normalize(A - A.mean(0, keepdim=True), dim=-1)
        Ln = self.llm_c[items]
        return ((An @ An.t() - Ln @ Ln.t()) ** 2).mean()

    def forward(self, seq, pos, neg, positions, **kwargs):
        d, parts = self.d, {}
        h, pi, ni = self._pairs(seq, pos, neg, positions)
        ep, en = self._get_embedding(pi), self._get_embedding(ni)
        p_co, p_se = (h[:, :d] * ep[:, :d]).sum(-1), (h[:, d:] * ep[:, d:]).sum(-1)
        n_co, n_se = (h[:, :d] * en[:, :d]).sum(-1), (h[:, d:] * en[:, d:]).sum(-1)
        if self.use_ngg:              # NGG: the negative's own collaborative embedding moves by beta_v
            beta = self.ngg_beta[ni]
            n_co_gated = (h[:, :d] * en[:, :d].detach()).sum(-1)
            n_co = beta * n_co + (1 - beta) * n_co_gated
        pos_logits, neg_logits = p_co + p_se, n_co + n_se
        loss = self.loss_func(pos_logits, torch.ones_like(pos_logits)) + \
            self.loss_func(neg_logits, torch.zeros_like(neg_logits))
        parts["rec"] = loss.item()
        if self.lambda_cv > 0:        # CCS, both directions
            w_se = torch.sigmoid(p_co - n_co).detach()
            l_se = (w_se * (F.softplus(-p_se) + F.softplus(n_se))).sum() / w_se.sum().clamp(min=1e-6)
            w_co = torch.sigmoid(p_se - n_se).detach()
            l_co = (w_co * (F.softplus(-p_co) + F.softplus(n_co))).sum() / w_co.sum().clamp(min=1e-6)
            loss = loss + self.lambda_cv * (l_se + l_co)
            parts["cv"] = l_se.item() + l_co.item()
        # LLM-ESR's retrieval-augmented self-distillation (LLMESR_<backbone>.forward); similar users are targets only
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        log_last = self._train_state(seq, positions)
        with torch.no_grad():
            sim_feats = self._train_state(sim_seq, sim_positions).view(seq.shape[0], sim_num, -1).mean(1)
        sd = self.align(log_last, sim_feats)
        loss = loss + self.alpha * sd
        parts["sd"] = sd.item()
        if self.lambda_sp > 0:        # IUSP
            lsp = self.sp_loss()
            loss = loss + self.lambda_sp * lsp
            parts["sp"] = lsp.item()
        self.parts = parts
        return loss

    @torch.no_grad()
    def item_tables(self):
        E = self._get_embedding(torch.arange(self.n_items + 2, device=self.dev))
        return E[:, :self.d], E[:, self.d:]


class CSB_SASRec(CSBMixin, LLMESR_SASRec):
    """Seq-to-seq: every non-padding position with its next item and one uniform negative."""

    def __init__(self, user_num, item_num, device, args):
        LLMESR_SASRec.__init__(self, user_num, item_num, device, args)
        self._csb_init(item_num, args)

    def _pairs(self, seq, pos, neg, positions):
        h = self.log2feats(seq, positions)
        idx = pos != 0
        return h[idx], pos[idx], neg[idx]

    def _train_state(self, seq, positions):
        return self.log2feats(seq, positions)[:, -1, :]

    @torch.no_grad()
    def user_reps(self, seq, positions):
        return self.log2feats(seq, positions)[:, -1, :]


class CSB_GRU4Rec(CSBMixin, LLMESR_GRU4Rec):
    """Seq-to-one: one pair per training sequence (its last item and one negative)."""

    def __init__(self, user_num, item_num, device, args):
        LLMESR_GRU4Rec.__init__(self, user_num, item_num, device, args)
        self._csb_init(item_num, args)

    def _pairs(self, seq, pos, neg, positions):
        h = self.log2feats(seq)[:, -1, :]
        idx = pos != 0
        return h[idx], pos[idx], neg[:, 0][idx]

    def _train_state(self, seq, positions):
        return self.log2feats(seq)[:, -1, :]

    @torch.no_grad()
    def user_reps(self, seq, positions):
        return self.log2feats(seq)[:, -1, :]


class CSB_Bert4Rec(CSBMixin, LLMESR_Bert4Rec):
    """Masked positions are the pairs; prediction appends a mask token, as Bert4Rec.predict."""

    def __init__(self, user_num, item_num, device, args):
        LLMESR_Bert4Rec.__init__(self, user_num, item_num, device, args)
        self._csb_init(item_num, args)

    def _pairs(self, seq, pos, neg, positions):
        m = pos > 0
        return self.log2feats(seq, positions)[m], pos[m], neg[m]

    def _train_state(self, seq, positions):
        return self.log2feats(seq, positions)[:, -1, :]

    @torch.no_grad()
    def user_reps(self, seq, positions):
        log_seqs = torch.cat([seq, torch.full_like(seq[:, :1], self.mask_token)], dim=1)[:, 1:]
        pos2 = torch.cat([positions, positions[:, -1:] + 1], dim=1)[:, 1:]
        return self.log2feats(log_seqs.long(), pos2.long())[:, -1, :]


BACKBONES = {"sasrec": CSB_SASRec, "gru4rec": CSB_GRU4Rec, "bert4rec": CSB_Bert4Rec}
