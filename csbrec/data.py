"""Data loading, evaluation candidates and metrics.

Training data come from the LLM-ESR generators (generators/), so the train / validation / test split, the similar users
of the retrieval-augmented self-distillation and the LLM embeddings are exactly those of LLM-ESR.
Evaluation ranks each target against 100 items the user never interacted with, drawn once per seed.
Item groups: a tail item has fewer than ts_item interactions; a tail user has a sequence shorter than ts_user
(the thresholds of LLM-ESR's experiments/<dataset>.bash).
"""
import json
import logging
import argparse
import numpy as np
import torch

from generators.generator import Seq2SeqGeneratorAllUser, GeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser

# (ts_user, ts_item) of LLM-ESR's experiments/<dataset>.bash
DATASET_TS = {"beauty": (9, 4), "fashion": (3, 4), "yelp": (12, 13)}


def silent_logger(name="csbrec.data"):
    lg = logging.getLogger(name)
    lg.handlers = []
    lg.addHandler(logging.NullHandler())
    lg.propagate = False
    return lg


def base_namespace(dataset, max_len=200, hidden_size=64, num_workers=8, train_batch_size=128, test_neg=100,
                   sim_user_num=10, **extra):
    """The fields the LLM-ESR generators and models read from main.py's arguments (experiments/<dataset>.bash)."""
    ns = argparse.Namespace(
        dataset=dataset, inter_file="inter", aug_file="inter", num_workers=num_workers,
        train_batch_size=train_batch_size, aug=False, aug_seq=False, aug_seq_len=0, max_len=max_len,
        train_neg=1, test_neg=test_neg, sim_user_num=sim_user_num,
        hidden_size=hidden_size, trm_num=2, num_heads=1, dropout_rate=0.5, use_cross_att=True, freeze=True,
        alpha=0.1, user_sim_func="kd", item_reg=False, beta=0.1)
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def make_generator(backbone, ns, device):
    """SASRec trains seq-to-seq, GRU4Rec seq-to-one and BERT4Rec on masked positions, as in LLM-ESR."""
    Gen = {"sasrec": Seq2SeqGeneratorAllUser, "gru4rec": GeneratorAllUser, "bert4rec": BertGeneratorAllUser}[backbone]
    gen = Gen(ns, silent_logger(), device)
    user_num, item_num = gen.get_user_item_num()
    return gen, user_num, item_num


def item_counts(dicts, item_num):
    """Interaction count of every item id (size item_num + 2: padding 0, items, mask token)."""
    items = [i for d in dicts for s in d.values() for i in s]
    return np.bincount(np.asarray(items, dtype=np.int64), minlength=item_num + 2)


def format_input(hist, max_len):
    """Right-aligned item ids and positions 1..len."""
    seq = np.zeros(max_len, dtype=np.int64)
    pos = np.zeros(max_len, dtype=np.int64)
    h = hist[-max_len:]
    if len(h) > 0:
        seq[max_len - len(h):] = h
        pos[max_len - len(h):] = np.arange(1, len(h) + 1)
    return seq, pos


def sample_candidates(seen_lists, targets, item_num, n_neg, rng):
    """Column 0 = target, then n_neg distinct unseen items.

    Reject impossible requests before sampling instead of looping indefinitely.
    The rejection-sampling order is unchanged for valid existing inputs.
    """
    if not isinstance(item_num, (int, np.integer)) or isinstance(item_num, (bool, np.bool_)) or item_num < 1:
        raise ValueError("item_num must be a positive integer")
    if not isinstance(n_neg, (int, np.integer)) or isinstance(n_neg, (bool, np.bool_)) or n_neg < 0:
        raise ValueError("n_neg must be a non-negative integer")
    if len(seen_lists) != len(targets):
        raise ValueError("seen_lists and targets must have the same length")
    exclusions = []
    for k, (seen, target) in enumerate(zip(seen_lists, targets)):
        if not isinstance(target, (int, np.integer)) or isinstance(target, (bool, np.bool_)) or not 1 <= target <= item_num:
            raise ValueError("target at row %d must be an integer in 1..item_num" % k)
        excl = set(seen)
        excl.add(target)
        available = item_num - sum(1 for item in excl if 1 <= item <= item_num)
        if n_neg > available:
            raise ValueError("row %d has only %d unseen items, but %d negatives were requested" % (k, available, n_neg))
        exclusions.append(excl)
    out = np.empty((len(targets), n_neg + 1), dtype=np.int64)
    out[:, 0] = targets
    for k, excl in enumerate(exclusions):
        negs = []
        while len(negs) < n_neg:
            for t in rng.integers(1, item_num + 1, size=2 * n_neg).tolist():
                if t not in excl:
                    excl.add(t)
                    negs.append(t)
                    if len(negs) == n_neg:
                        break
        out[k, 1:] = negs
    return out


class Split:
    def __init__(self, users, hists, targets, cand, tail_item, short_user, max_len):
        rows = [format_input(h, max_len) for h in hists]
        self.users, self.targets = users, np.asarray(targets, dtype=np.int64)
        self.seq = torch.from_numpy(np.stack([r[0] for r in rows]))
        self.pos = torch.from_numpy(np.stack([r[1] for r in rows]))
        self.cand = torch.from_numpy(cand)
        self.tail_item, self.short_user = tail_item, short_user
        self.masks = {"overall": np.ones(len(hists), dtype=bool), "tail_item": tail_item, "head_item": ~tail_item,
                      "tail_user": short_user, "head_user": ~short_user}
        self.n = len(hists)


def build_eval(gen, item_num, seed, ts_user, ts_item, max_len, test_neg=100):
    """Validation input = training sequence; test input = training + validation items."""
    train, valid, test = gen.train, gen.valid, gen.test
    users = [u for u in train if len(valid[u]) > 0]
    pop_all = item_counts([train, valid, test], item_num)
    rng = np.random.default_rng(seed)
    out = {}
    for split in ["valid", "test"]:
        if split == "valid":
            hists = [train[u] for u in users]
            tgts = [valid[u][0] for u in users]
            seen = [train[u] + valid[u] for u in users]
        else:
            hists = [train[u] + valid[u] for u in users]
            tgts = [test[u][0] for u in users]
            seen = [train[u] + valid[u] + test[u] for u in users]
        cand = sample_candidates(seen, tgts, item_num, test_neg, rng)
        seqlen = np.minimum(np.array([len(h) for h in hists]), max_len)
        out[split] = Split(users, hists, tgts, cand, pop_all[np.array(tgts)] < ts_item, seqlen < ts_user, max_len)
    return out, pop_all


def ranks(s):
    """Rank of column 0 among the candidates; ties count against the target."""
    return (s[:, 1:] >= s[:, :1]).sum(axis=1)


def metrics(rk, masks, k=10):
    hit = rk < k
    ndcg = np.where(hit, 1.0 / np.log2(rk + 2.0), 0.0)
    out = {}
    for g, m in masks.items():
        out["%s_N@%d" % (g, k)] = float(ndcg[m].mean()) if m.any() else 0.0
        out["%s_H@%d" % (g, k)] = float(hit[m].mean()) if m.any() else 0.0
    return out


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
