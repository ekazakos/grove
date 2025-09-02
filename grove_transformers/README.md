This package is a implemented with 🤗 Transformers and provides a lightweight, inference-only interface for GROVE.

---

## Installation

Install the inference package:

If you've already cloned the main repo from [https://github.com/ekazakos/grove/](https://github.com/ekazakos/grove/), then run:
```bash
cd grove/grove_transformers
pip install -e .[torch] --extra-index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.3 --no-build-isolation
```


Alternatively, run:
```bash
pip install -e "git+https://github.com/ekazakos/grove.git#subdirectory=grove_transformers[torch]" \
  --extra-index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.3 --no-build-isolation
```

Also, install **mmcv**, **mmdetection** and **SAM2** as shown [here](https://github.com/ekazakos/grove?tab=readme-ov-file#install-mmdetection).

### Alternative (quick hack)

If you already cloned the full repo and want to avoid installing the package, you can make it importable by setting:

```bash
export PYTHONPATH=/path/to/grove/grove_transformers
```

---

## Notes
- This model requires Python ≥3.11.
- Auto* classes (e.g. `AutoTokenizer`) are **not supported**; use the custom `Grove*` classes.

---

## Example Usage 1: Minimal (automatic metadata)

If you don’t have precomputed token embeddings for GROVE's vocabulary or video metadata, just pass the video path.  
GROVE will compute everything internally.

```python
from grove_transformers import GroveTokenizer, GroveForCausalLM, GroveProcessor

tokenizer = GroveTokenizer.from_pretrained("ekazakos/grove")
model = GroveForCausalLM.from_pretrained(
    "ekazakos/grove",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)
processor = GroveProcessor.from_pretrained("ekazakos/grove")

outputs = processor.generate(
    model,
    video_path,
    token_embeddings=None,
    device="cuda",
    start_frame=None,
    end_frame=None,
    video_width=None,
    video_height=None,
    video_fps=None
)
```

---

## Example Usage 2: With precomputed inputs

If you have precomputed token embeddings for GROVE's vocabulary and video metadata (e.g. from datasets like **HowToGround1M** or **iGround**), you can pass them directly for faster inference and precise trimming.

```python
outputs = processor.generate(
    model,
    video_path,
    token_embeddings=precomputed_embeddings,
    device="cuda",
    start_frame=dataset_meta["start_frame"],
    end_frame=dataset_meta["end_frame"],
    video_width=dataset_meta["width"],
    video_height=dataset_meta["height"],
    video_fps=dataset_meta["fps"]
)
```

---

## Notes

- **`token_embeddings`**: pass precomputed token embeddings for speed, or `None` to compute on the fly.  
  For precomputing token embeddings for GROVE's vocabulary, see [embed_tokens.sh](https://github.com/ekazakos/grove/blob/main/embed_tokens.sh).  
- **Video metadata** (`start_frame`, `end_frame`, `video_width`, `video_height`, `video_fps`): pass if available, otherwise `None` → GROVE computes automatically.  
- **Trimming**: `start_frame`/`end_frame` let you process only part of a video.  

---

```bibtex
@article{kazakos2025grove,
  title={Large-scale Pre-training for Grounded Video Caption Generation},
  author={Evangelos Kazakos and Cordelia Schmid and Josef Sivic},
  journal={arXiv preprint arXiv:2503.10781},
  year={2025}
}
```