# grove_new

Legacy GROVE refactor scaffold (faithful reproduction adapted to HuggingFace patterns).

Components:
- GroveConfig: extends LlamaConfig with GROVE / SAM / loss parameters.
- GroveForCausalLM: wraps legacy GROVE model pathways (training vs inference), SAM grounding encoder, text projection layer.
- GroveTokenizer / GroveProcessor: placeholders pending migration of special token handling.

Differences vs original:
- Centralizes configuration in GroveConfig (no ad-hoc attribute inserts outside init).
- Loss weighting pulled from config/kwargs and stored as module attributes.
- Device moves dynamic (no hard-coded .cuda()).
- Detached experimental CLIP fusion prototype (not used; staying with SAM + LLaMA per legacy directory).

Next steps (legacy alignment):
1. Port evaluate() method into GroveForCausalLM (optional if needed for generation-first pass).
2. Migrate tokenizer special tokens & indices from legacy tokenizer usage.
3. Provide checkpoint weight mapping script (old key -> new module paths).
4. Add tests: model_forward (train), inference (inference=True), get_grounding_encoder_embs.
5. Integrate process for dense_pe retrieval through standard forward utilities.
