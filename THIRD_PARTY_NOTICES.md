# Third-party notices

CSBRec builds on the MIT-licensed projects listed below. Their licence texts are reproduced below.

## LLM-ESR

Source: https://github.com/Applied-Machine-Learning-Lab/LLM-ESR

Used in this repository: `main.py`, `models/`, `generators/`, `trainers/`, `utils/`, `data/` and `experiments/`.
The CSBRec sub-models in `csbrec/model.py` subclass LLM-ESR's dual-view models.

Local changes to the upstream utilities and preprocessing include:
- `np.Inf` replaced by `np.inf` in `utils/earlystop.py` for NumPy 2 compatibility.
- Metadata parsing uses JSON or `ast.literal_eval` rather than `eval`.
- Embedding notebooks read API settings from environment variables, use request timeouts and bounded retries,
  and save completed embeddings before propagating a failure or interruption. The embedding model remains
  `text-embedding-ada-002` and the original prompt truncation rules are retained.
- Removed hard-coded exploratory item lookups, notebook outputs/execution counts, and obsolete machine-path
  comments; corrected the retrieval notebook's default dataset directory from `yelp_new` to `yelp`.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## FCRLLM

Source: https://github.com/Byungmoon-Heo/FCRLLM

Used in this repository: the flipped-classroom framework in `csbrec/flipclass.py` (Hopfield update, attention
blocks, ST / CT alignment and score averaging), adapted from FCRLLM's `models/FlipClassroom.py` and
`models/utils.py`.

```
MIT License

Copyright (c) 2026 Byungmoon-Heo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## early-stopping-pytorch

Source: https://github.com/Bjarten/early-stopping-pytorch

Used in this repository: `utils/earlystop.py`, carried through LLM-ESR and adapted to stop on validation
performance. The original attribution remains in that file.

```
MIT License

Copyright (c) 2018 Bjarte Mehus Sunde

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
