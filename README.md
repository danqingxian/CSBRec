# CSBRec

Code for the paper **"CSBRec: Mitigating Collaborative–Semantic Learning Imbalance for LLM-Enhanced Long-Tail Sequential Recommendation"**.

CSBRec (Collaborative–Semantic Balanced Recommendation) trains a dual-view sequential recommender, in which every item has a trainable collaborative ID embedding and a frozen LLM semantic embedding mapped by an adapter, so that the two views learn long-tail items in a balanced way. It is built on the flipped-classroom framework and keeps LLM-ESR's retrieval-augmented self-distillation. Three training modules address three imbalances of dual-view joint training:

| Module | Imbalance | What it does |
| --- | --- | --- |
| **NGG** Negative-Gradient Gating | negative gradients dominate the collaborative embeddings of long-tail items | scales the gradient that a sampled negative sends into its own collaborative embedding by a gate β_v (β_v = 0 for items that are never a training positive) |
| **CCS** Cross-view Compensating Supervision | gradient starvation of the semantic view | gives each view its own BCE loss on the same pairs, weighted by how well the other view alone already separates the pair |
| **IUSP** Item-Uniform Semantic structure Preservation | dimensional collapse of the adapter output | on items drawn uniformly from the item set, matches the cosine-similarity structure of the adapter outputs to that of the LLM embeddings |

All three modules act on the training loss only. At inference CSBRec scores items exactly like the base model, so the network structure and the inference cost stay the same.

Per sub-model loss:

```
L_CSB = L_rec^β + α·L_SD + λ_cv·L_cv + λ_sp·L_sp
```

Flipped-classroom framework, with the two parameter-independent sub-models ST (semantic view as teacher) and CT (collaborative view as teacher):

```
L = 1/2 (L_CSB^ST + λ_fc·L_align^ST) + 1/2 (L_CSB^CT + λ_fc·L_align^CT)
```

At inference the scores of the two sub-models are averaged.

## Repository structure

```
csbrec/
  model.py        CSBRec sub-models on SASRec, GRU4Rec and BERT4Rec (NGG, CCS, IUSP)
  flipclass.py    flipped-classroom framework (two sub-models, Hopfield alignment, score averaging)
  data.py         data loading, evaluation candidates, metrics
  train.py        training and evaluation entry point
scripts/
  run_main.sh         3 backbones x 3 datasets x 3 seeds
  run_ablation.sh     module ablation (SASRec)
  run_sensitivity.sh  hyper-parameter sensitivity (SASRec)
models/ generators/ trainers/ utils/ main.py experiments/   LLM-ESR code (base models, data pipeline, LLM-ESR runs)
data/             data preparation scripts and notebooks from LLM-ESR
```

## Environment

The experiments were run with Python 3.12, PyTorch 2.5.1 (CUDA 12.4) and NumPy 2.1 on a single NVIDIA RTX 4090. The requirements below give minimum versions.

```bash
pip install -r requirements.txt
```

Training holds two dual-view sub-models, so one run needs up to about 16 GB (SASRec), 18 GB (GRU4Rec) or 22 GB (BERT4Rec) of GPU memory.

## Data preparation

CSBRec uses the datasets, preprocessing and LLM embeddings of [LLM-ESR](https://github.com/Applied-Machine-Learning-Lab/LLM-ESR): Yelp, Amazon Fashion and Amazon Beauty. Each dataset needs five files in `data/<yelp|fashion|beauty>/handled/`:

| File | Content |
| --- | --- |
| `inter.txt` | user–item interaction sequences |
| `itm_emb_np.pkl` | LLM item embeddings |
| `usr_emb_np.pkl` | LLM user embeddings |
| `pca64_itm_emb_np.pkl` | 64-dimensional PCA of the item embeddings |
| `sim_user_100.pkl` | similar users for the retrieval-augmented self-distillation |

**Option 1: download the preprocessed files.** The LLM-ESR authors provide all of them in one archive (link in the [LLM-ESR README](https://github.com/Applied-Machine-Learning-Lab/LLM-ESR#preprocess-the-dataset)). Unpack it so that the files sit in `data/<dataset>/handled/`.

**Option 2: build them from the raw data**, following LLM-ESR:

1. Download the raw data into `data/<dataset>/raw/`: Yelp from <https://www.yelp.com/dataset>, Fashion and Beauty from the [Amazon review data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews).
2. Run `data/data_process.py` to filter users and items, then `data/convert_inter.ipynb` to write `inter.txt`.
3. Run `data/<dataset>/get_item_embedding.ipynb` and `data/<dataset>/get_user_embedding.ipynb` to obtain the LLM embeddings. Before starting the notebook kernel, set `CSBREC_EMBEDDING_URL` to your full embeddings endpoint and `CSBREC_EMBEDDING_API_KEY` to your API key. The code adds the `Bearer` prefix. The notebooks retain the upstream `text-embedding-ada-002` model setting, use a 60-second timeout and retry transient errors at most three times. They save completed requests to the cache before propagating a failure or interruption.
4. Run `data/pca.ipynb` for `pca64_itm_emb_np.pkl` and `data/retrieval_users.ipynb` for `sim_user_100.pkl`.

Run `data/data_process.py` from the repository root. Use `data/` as the working directory for its top-level notebooks, and `data/<dataset>/` for the embedding notebooks; select the intended `dataset` in each notebook. A notebook interface such as Jupyter is an optional extra installation.

## Training and evaluation

Run every command from the repository root.

One run:

```bash
python -m csbrec.train --backbone sasrec --dataset yelp --seed 42 --out runs/sasrec/yelp/s42
```

`--backbone` is `sasrec`, `gru4rec` or `bert4rec`; `--dataset` is `yelp`, `fashion` or `beauty`. The output directory receives `train.log`, `epochs.jsonl` (per-epoch losses and validation metrics), `best.pt` (best checkpoint without the frozen LLM table) and `result.json` (validation and test metrics at the best epoch).

All main experiments, the ablation and the sensitivity analysis:

```bash
bash scripts/run_main.sh        # optional argument: GPU id
bash scripts/run_ablation.sh
bash scripts/run_sensitivity.sh
```

LLM-ESR itself can be run with LLM-ESR's original scripts, for example `bash experiments/yelp.bash`.

On Windows, the Python command can run in PowerShell; the `.sh` and `.bash` launchers require Bash (for example Git Bash or WSL). Start with `--num_workers 0` when diagnosing data-loader problems. Run the candidate-sampling regression checks with:

```bash
python -m unittest discover -s tests -v
```

### Protocol

Adam with learning rate 1e-3, batch size 128, maximum sequence length 200, embedding size 64, at most 200 epochs and early stopping on validation NDCG@10 with patience 20. Every test user's target item is ranked against 100 items the user never interacted with, drawn once per seed. The reported metrics are NDCG@10 and HR@10 for all users, for tail and head items and for tail and head users. The group thresholds follow LLM-ESR: items with fewer than 13 / 4 / 4 interactions are tail items on Yelp / Fashion / Beauty, and users with sequences shorter than 12 / 3 / 9 items are tail users.

### Hyper-parameters

| Argument | Paper symbol | Default |
| --- | --- | --- |
| `--ngg binary` | κ → ∞ | NGG gate β_v = 1[v is a training positive] |
| `--ngg ratio --kappa K` | κ | β_v = min(1, κ·P_v / P̄) |
| `--ngg none` | | NGG off |
| `--lambda_cv` | λ_cv | 0.25 (0 turns CCS off) |
| `--lambda_sp` | λ_sp | 1.0 (0 turns IUSP off) |
| `--sp_items` | B | 1024 |
| `--lambda_fc` | λ_fc | per backbone and dataset, see below |
| `--mu`, `--rho`, `--eta` | μ, ρ, η | per backbone and dataset, see below |

The self-distillation weight α is 0.1, as in LLM-ESR. The flipped-classroom settings (λ_fc, μ, ρ, η) default to the values below, taken over from FCRLLM, and can be overridden on the command line:

| Backbone | Yelp | Fashion | Beauty |
| --- | --- | --- | --- |
| SASRec | 0.1, 0.5, 0.3, 0.7 | 0.5, 0.7, 0.7, 0.3 | 1.0, 0.3, 0.5, 0.7 |
| GRU4Rec | 1.0, 0.5, 0.3, 0.3 | 0.5, 0.7, 0.3, 0.5 | 0.5, 0.3, 0.3, 0.3 |
| BERT4Rec | 1.0, 0.7, 0.3, 0.5 | 1.0, 0.7, 0.3, 0.5 | 1.0, 0.7, 0.3, 0.5 |

## Acknowledgements

This code builds on [LLM-ESR](https://github.com/Applied-Machine-Learning-Lab/LLM-ESR) (base dual-view models, data pipeline, self-distillation) and [FCRLLM](https://github.com/Byungmoon-Heo/FCRLLM) (flipped-classroom framework). Both are released under the MIT licence; their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{csbrec,
  title   = {CSBRec: Mitigating Collaborative--Semantic Learning Imbalance for LLM-Enhanced Long-Tail Sequential Recommendation},
  author  = {},
  journal = {},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
