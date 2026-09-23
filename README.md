# Modified-Ai-training-Lib

## Vendored Hugging Face Transformers

`transformers/` contains the full source of [huggingface/transformers](https://github.com/huggingface/transformers)
**v5.17.0** (upstream commit `856157a2f3e9594954310df18fdccc31ffddebe9`), licensed under Apache-2.0
(see `transformers/LICENSE`). It is vendored here so it can be modified directly.

### Setup (editable install)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch
pip install -e ./transformers
python -c "import transformers; print(transformers.__version__, transformers.__file__)"
```

Edits under `transformers/src/transformers/` take effect immediately. Existing hybrid
Transformer/Mamba references: `models/jamba`, `models/bamba`, `models/zamba2`, `models/falcon_h1`,
`models/granitemoehybrid`, `models/mamba2`.
