"""Train and evaluate CSBRec on one backbone, dataset and seed.

Protocol: Adam (lr 1e-3, no weight decay), batch 128, maximum sequence length 200, at most 200 epochs, early stopping on
validation overall NDCG@10 with patience 20; every user's target is ranked against 100 items the user never interacted
with, drawn once from the seed. Reported metrics: NDCG@10 and HR@10 for all users and for tail / head items and users.

Usage:
  python -m csbrec.train --backbone sasrec --dataset yelp --seed 42 --out runs/sasrec/yelp/s42
Ablation and sensitivity:
  --ngg none | --lambda_cv 0 | --lambda_sp 0          remove NGG / CCS / IUSP
  --ngg ratio --kappa 4 | --lambda_cv 0.5 | --lambda_sp 3   vary one module
The flipped-classroom settings (lambda_fc, mu, rho, eta) default to the values in FLIP_DEFAULTS for the backbone and
dataset; any of them can be overridden on the command line.
"""
import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import torch

from utils.utils import set_seed
from csbrec.data import DATASET_TS, base_namespace, make_generator, build_eval, ranks, metrics, save_json
from csbrec.model import BACKBONES
from csbrec.flipclass import CSBRec

# flipped-classroom settings (lambda_fc, mu, rho, eta) per backbone and dataset
FLIP_DEFAULTS = {
    ("sasrec", "yelp"): (0.1, 0.5, 0.3, 0.7),
    ("sasrec", "fashion"): (0.5, 0.7, 0.7, 0.3),
    ("sasrec", "beauty"): (1.0, 0.3, 0.5, 0.7),
    ("gru4rec", "yelp"): (1.0, 0.5, 0.3, 0.3),
    ("gru4rec", "fashion"): (0.5, 0.7, 0.3, 0.5),
    ("gru4rec", "beauty"): (0.5, 0.3, 0.3, 0.3),
    ("bert4rec", "yelp"): (1.0, 0.7, 0.3, 0.5),
    ("bert4rec", "fashion"): (1.0, 0.7, 0.3, 0.5),
    ("bert4rec", "beauty"): (1.0, 0.7, 0.3, 0.5),
}


def get_args(argv=None):
    p = argparse.ArgumentParser(description="CSBRec training")
    p.add_argument("--backbone", required=True, choices=list(BACKBONES))
    p.add_argument("--dataset", required=True, choices=list(DATASET_TS))
    p.add_argument("--out", required=True, help="output directory (train.log, epochs.jsonl, best.pt, result.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_train_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval_batch", type=int, default=1024)
    p.add_argument("--mask_prob", type=float, default=0.6, help="BERT4Rec masking probability")
    # NGG
    p.add_argument("--ngg", default="binary", choices=["binary", "ratio", "none"],
                   help="binary: kappa -> inf (default); ratio: beta_v = min(1, kappa * P_v / P_mean); none: off")
    p.add_argument("--kappa", type=float, default=1.0, help="NGG kappa for --ngg ratio")
    # CCS
    p.add_argument("--lambda_cv", type=float, default=0.25, help="CCS weight (0 = off)")
    # IUSP
    p.add_argument("--lambda_sp", type=float, default=1.0, help="IUSP weight (0 = off)")
    p.add_argument("--sp_items", type=int, default=1024, help="IUSP items drawn per step (B)")
    # flipped classroom (None = FLIP_DEFAULTS)
    p.add_argument("--lambda_fc", type=float, default=None, help="alignment loss weight (FCRLLM gamma_fc)")
    p.add_argument("--mu", type=float, default=None, help="Hopfield inverse temperature (FCRLLM beta_hopfield)")
    p.add_argument("--rho", type=float, default=None, help="Hopfield regularisation (FCRLLM alpha_hopfield)")
    p.add_argument("--eta", type=float, default=None, help="Hopfield step size (FCRLLM gamma_update)")
    args = p.parse_args(argv)
    for name in ("max_len", "num_train_epochs", "patience", "eval_batch"):
        if getattr(args, name) <= 0:
            p.error("--%s must be positive" % name)
    if args.num_workers < 0:
        p.error("--num_workers must be nonnegative")
    if not np.isfinite(args.lr) or args.lr <= 0:
        p.error("--lr must be finite and positive")
    if not 0 < args.mask_prob <= 1:
        p.error("--mask_prob must be in (0, 1]")
    if args.ngg == "ratio" and (not np.isfinite(args.kappa) or args.kappa <= 0):
        p.error("--kappa must be finite and positive")
    for name in ("lambda_cv", "lambda_sp", "lambda_fc"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            p.error("--%s must be finite and nonnegative" % name)
    if args.lambda_sp > 0 and args.sp_items < 2:
        p.error("--sp_items must be at least 2 when IUSP is enabled")
    dflt = dict(zip(["lambda_fc", "mu", "rho", "eta"], FLIP_DEFAULTS[(args.backbone, args.dataset)]))
    for k, v in dflt.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def make_logger(path):
    lg = logging.getLogger("csbrec.train")
    lg.setLevel(logging.INFO)
    lg.handlers = []
    fm = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")
    for h in [logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        h.setFormatter(fm)
        lg.addHandler(h)
    lg.propagate = False
    return lg


def ngg_beta(backbone, gen, item_num, mode, kappa):
    """P_v = number of times item v is a training positive under the backbone's objective: SASRec every next item of a
    training sequence, GRU4Rec and BERT4Rec every training occurrence (BERT4Rec can mask any position)."""
    if backbone == "sasrec":
        items = [i for u in gen.train for i in gen.train[u][1:]]
    else:
        items = [i for u in gen.train for i in gen.train[u]]
    P = np.bincount(np.asarray(items, dtype=np.int64), minlength=item_num + 2)[1:item_num + 1].astype(np.float64)
    if mode == "ratio":
        return np.minimum(1.0, kappa * P / (P.sum() / item_num))
    return (P > 0).astype(np.float64)


@torch.no_grad()
def scores(model, seq, pos, cand, device, bs):
    """Candidate scores s = s_co + s_se (each view averaged over the two sub-models)."""
    model.eval()
    E_co, E_se = model.item_tables()
    d = model.d
    out = np.zeros(cand.shape, dtype=np.float32)
    for st in range(0, len(seq), bs):
        h = model.user_reps(seq[st:st + bs].to(device), pos[st:st + bs].to(device))
        c = cand[st:st + bs].to(device)
        co = (E_co[c] * h[:, None, :d]).sum(-1)
        se = (E_se[c] * h[:, None, d:]).sum(-1)
        out[st:st + bs] = (co.cpu().numpy() + se.cpu().numpy())
    return out


def build(args, device):
    ns = base_namespace(args.dataset, max_len=args.max_len, num_workers=args.num_workers,
                        ngg=args.ngg, lambda_cv=args.lambda_cv, lambda_sp=args.lambda_sp, sp_items=args.sp_items,
                        lambda_fc=args.lambda_fc, mu=args.mu, rho=args.rho, eta=args.eta)
    if args.backbone != "sasrec":
        ns.mask_prob, ns.num_layers = args.mask_prob, 1
    gen, user_num, item_num = make_generator(args.backbone, ns, device)
    model = CSBRec(BACKBONES[args.backbone], user_num, item_num, device, ns).to(device)
    info = None
    if args.ngg != "none":
        beta = ngg_beta(args.backbone, gen, item_num, args.ngg, args.kappa)
        model.set_ngg_beta(beta)
        info = {"items_with_beta_0": int((beta == 0).sum()), "mean_beta": float(beta.mean())}
    return ns, gen, user_num, item_num, model, info


def main(argv=None):
    args = get_args(argv)
    data_dir = os.path.join("data", args.dataset, "handled")
    required = ("inter.txt", "itm_emb_np.pkl", "pca64_itm_emb_np.pkl", "sim_user_100.pkl")
    missing = [name for name in required if not os.path.isfile(os.path.join(data_dir, name))]
    if missing:
        raise SystemExit("Missing prepared data in %s: %s. Run from the repository root and follow "
                         "README.md, Data preparation." % (data_dir, ", ".join(missing)))
    os.makedirs(args.out, exist_ok=True)
    lg = make_logger(os.path.join(args.out, "train.log"))
    lg.info("CSBRec %s on %s, seed %d: %s" % (args.backbone, args.dataset, args.seed, json.dumps(vars(args))))
    ts_user, ts_item = DATASET_TS[args.dataset]
    set_seed(args.seed)
    device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    ns, gen, user_num, item_num, model, info = build(args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loader = gen.make_trainloader()
    var_name = gen.train_dataset.var_name
    splits, _ = build_eval(gen, item_num, args.seed, ts_user, ts_item, args.max_len)
    if info:
        lg.info("NGG %s: %s" % (args.ngg, info))
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lg.info("users %d, items %d, trainable parameters %d" % (user_num, item_num, n_par))

    ev = splits["valid"]
    best, best_epoch, bad, epoch = None, -1, 0, -1
    ckpt = os.path.join(args.out, "best.pt")
    flog = open(os.path.join(args.out, "epochs.jsonl"), "w")
    t_start = time.time()
    for epoch in range(args.num_train_epochs):
        t0 = time.time()
        model.train()
        sums, steps = {}, 0
        for batch in loader:
            inputs = dict(zip(var_name, [t.to(device) for t in batch]))
            opt.zero_grad()
            lv = model.step_backward(inputs)
            opt.step()
            if lv == lv:   # a batch without positives gives a NaN loss value (empty mean), as in LLM-ESR
                sums["loss"] = sums.get("loss", 0.0) + lv
                for k, v in model.parts.items():
                    sums[k] = sums.get(k, 0.0) + v
                steps += 1
        m = metrics(ranks(scores(model, ev.seq, ev.pos, ev.cand, device, args.eval_batch)), ev.masks)
        score = m["overall_N@10"]
        flag = ""
        if best is None or score > best:
            best, best_epoch, bad = score, epoch, 0
            sd = {k: v for k, v in model.state_dict().items() if not k.endswith("llm_item_emb.weight")}
            torch.save({"epoch": epoch, "state_dict": sd}, ckpt)
            flag = " *"
        else:
            bad += 1
        rec = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in sums.items()}, "valid": m, "time_s": time.time() - t0}
        flog.write(json.dumps(rec) + "\n")
        flog.flush()
        lg.info("[epoch %d] loss %.4f | valid N@10 overall %.4f tail item %.4f head item %.4f | %.0fs%s" % (
            epoch, rec.get("loss", float("nan")), score, m["tail_item_N@10"], m["head_item_N@10"], rec["time_s"], flag))
        if bad >= args.patience:
            break
    flog.close()
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)   # the frozen LLM table is rebuilt from the data
    res = {"config": vars(args), "ngg": info, "best_epoch": best_epoch, "epochs_run": epoch + 1,
           "train_time_s": time.time() - t_start}
    for split in ["valid", "test"]:
        e = splits[split]
        res[split] = metrics(ranks(scores(model, e.seq, e.pos, e.cand, device, args.eval_batch)), e.masks)
    r = res["test"]
    lg.info("[test, best epoch %d] N@10 overall %.4f tail item %.4f head item %.4f tail user %.4f head user %.4f | "
            "H@10 overall %.4f tail item %.4f" % (best_epoch, r["overall_N@10"], r["tail_item_N@10"], r["head_item_N@10"],
                                                  r["tail_user_N@10"], r["head_user_N@10"], r["overall_H@10"],
                                                  r["tail_item_H@10"]))
    save_json(res, os.path.join(args.out, "result.json"))
    lg.info("done in %.0fs" % (time.time() - t_start))


if __name__ == "__main__":
    main()
