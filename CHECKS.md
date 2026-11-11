# Release check — 2026-11-12

This is a pre-submission working snapshot. The checks below establish basic execution and packaging status; they do not reproduce the manuscript's reported scores. Changes to the scientific training protocol were kept separate from this cleanup.

## Checks completed

- No actual API keys, private credentials or saved notebook outputs were found in the release files. Raw datasets, embeddings, training logs and checkpoints are excluded from Git.
- Python source and all 118 code cells in nine notebooks passed syntax checks.
- Six candidate-sampling regression checks passed, including impossible requests, target exclusion and unchanged seeded sampling for valid existing inputs.
- SASRec, GRU4Rec and BERT4Rec passed synthetic forward/backward, finite-gradient, sequential sub-model backward, 101-candidate scoring, score-averaging and checkpoint-reload checks. Test environment: Python 3.11.9, PyTorch 2.8.0+cu126, NumPy 2.3.3 and pandas 2.3.3; model checks ran on CPU.
- The six embedding helpers passed 30 mocked request scenarios and 12 cache-on-failure/interruption checks. No live embedding requests were made.
- Third-party MIT notices were checked against the upstream projects, including the early-stopping utility. See `THIRD_PARTY_NOTICES.md`.

## Corrections included

- Candidate sampling now fails with a useful message when the catalogue cannot supply enough distinct negatives, instead of looping indefinitely. The target is explicitly excluded from negatives.
- Added the missing `tensorboard` dependency for the inherited LLM-ESR entry point; removed an unverified exact-environment claim from the dependency file.
- Added training-argument checks and an early missing-data message before creating run outputs.
- Embedding notebooks use environment variables, timeouts, bounded retries and cache preservation. Notebook execution counters and outputs have been cleared.
- Replaced executable `eval` parsing of downloaded metadata with JSON / `ast.literal_eval`, removed upstream machine-specific path comments and corrected the retrieval notebook's default dataset directory.
- Added credential exclusions and explicit text line endings for cross-platform use.

## Scientific issues requiring reconciliation

### 1. SASRec's sequential training path can read later positions

`models/SASRec.py` creates a causal attention mask but does not pass it to `MultiheadAttention`. The cross-view attention in `models/utils.py` also masks padding only. For the sequence-to-sequence next-item objective, later input positions can therefore contain the positive target of an earlier position.

This was reproduced with dropout disabled by evaluation mode: changing only the sixth input item while leaving the first five unchanged changed the representation at the first position (maximum absolute difference 3.18989849 in the synthetic check). This demonstrates future-position visibility within a training sequence; it does not establish that test examples were read during training.

Both attention paths need a deliberate causality review. Correcting them changes the training protocol, so relevant baselines and CSBRec results must be reevaluated before asserting that the manuscript's comparisons remain valid. No attention behavior was changed in this cleanup.

### 2. IUSP uses different centering references in code and manuscript

In `csbrec/model.py`, `_csb_init` centers the frozen LLM embeddings using the full item catalogue mean. `sp_loss` centers adapter outputs using the sampled batch mean. The supplied manuscript, Section 4.3 / equation 22, describes batch means for both spaces.

The actual experimental implementation must determine whether the manuscript should be corrected or the implementation changed and experiments rerun. The cleanup preserved the supplied model behavior.

### 3. NGG counts have a backbone-specific convention

`ngg_beta` counts every item after the first item of each SASRec training sequence, before `max_len` truncation. GRU4Rec and BERT4Rec count all training occurrences, which the manuscript's experimental settings explicitly allow. These counts should not be described uniformly as the number of realized training targets. The SASRec truncation convention still needs to be reconciled with any interpretation based on actual sampled targets.

### 4. The package covers only part of the manuscript's experimental evidence

The current entry point runs training and fixed 100-negative sampled evaluation. No entry points were found for the manuscript's full-catalogue ranking table, diagnostic probes or mechanism figures. Real processed data, frozen embeddings, trained checkpoints, per-seed result logs and statistical-test inputs are not included.

The original preprocessing notebooks also require external raw data and API access. Synthetic checks do not establish end-to-end raw-data reproducibility, full-run GPU memory use or agreement with the manuscript's numerical results.

## Submission status

This snapshot is suitable for private version tracking while the issues above are resolved. Before a public research release or submission, reconcile the training protocol and IUSP definition, provide the missing experiment entry points/evidence, complete citation metadata and arrange reviewer access to code and data.
