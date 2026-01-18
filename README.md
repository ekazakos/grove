# 🚀 Large-scale Pre-training for Grounded Video Caption Generation

**Evangelos Kazakos<sup>1</sup>, Cordelia Schmid<sup>2</sup>, Josef Sivic<sup>1</sup>**  
<sup>1</sup>Czech Institute of Informatics, Robotics and Cybernetics at the Czech Technical University in Prague  
<sup>2</sup>Inria, École normale supérieure, CNRS, PSL Research University

📄 [**arXiv**](https://arxiv.org/abs/2503.10781) | 🌐 [**Project Website**](https://ekazakos.github.io/grounded_video_caption_generation/)

![Project Banner](teaser.png)

### 📢 News
- 🗂️ **18/01/2027**: We updated the dataset download process. You can now download the videos of our **HowToGround1M**, but also those of the original **HowTo100M** in addition to **iGround**. See the updated instructions in the README.
- 📚 **09/11/2025**: The **HowToGround1M** and **iGround** datasets are now available on 🤗 Hugging Face: [HowToGround1M](https://huggingface.co/datasets/ekazakos/HowToGround1M) | [iGround](https://huggingface.co/datasets/ekazakos/iGround). They can be loaded directly with `load_dataset()` from the 🤗 Datasets library.
- 🤗 **02/09/2025**: We release **grove-transformers** — a lightweight, inference-only interface for GROVE, implemented with 🤗 Transformers.
- 💻 **21/08/2025**: Code, checkpoints, and datasets released!
- 🔥 **25/06/2025**: Paper accepted to **ICCV 2025** 🎉

---

📖 **BibTeX**
```bibtex
@inproceedings{kazakos2025grove,
  title     = {Large-scale Pre-training for Grounded Video Caption Generation},
  author    = {Evangelos Kazakos and Cordelia Schmid and Josef Sivic},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

## Quick start

If you only need **inference for GROVE** (quick experimentation, no training), use the lightweight [`grove-transformers`](./grove_transformers) package instead. Installation instructions are in its [README](./grove_transformers/README.md) and on the [Hugging Face Hub](https://huggingface.co/ekazakos/grove).

If you want to **train and/or evaluate GROVE** end-to-end, follow the instructions below.

---

## Installation instructions

### Create and activate a pip environment
First, create a new conda environment:
```bash
conda create -n grove python=3.12
conda activate grove
```

---

### Install PyTorch
Choose the **CUDA version** that matches your system (e.g., `cu124`, `cu121`, `cu118`).  
Example for CUDA **12.4**:
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install torchtext==0.18.0 torchdata==0.9.0
```
> 💡 Replace `cu124` in the URL with the correct CUDA version tag for your machine.

---

### Install **mmdetection**
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d
pip install -e . --no-build-isolation
cd ..
```

### Install **mmcv**
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout 57c4e25e06e2d4f8a9357c84bcd24089a284dc88
pip install -r requirements/optional.txt
pip install -e . -v
cd ..
```

### Install **SAM2**
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
pip install -e . --no-build-isolation
cd ..
```

### Install **Flash Attention**
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

---

### Download Stanford CoreNLP

- [Stanford CoreNLP 3.4.1](https://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip) (for evaluation in HowToGround1M/iGround)

- [Stanford CoreNLP 4.5.7](https://nlp.stanford.edu/software/stanford-corenlp-4.5.7.zip) (for evaluation in ActivityNet-Entities)

---

### Install remaining dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation

### HowToGround1M / iGround / HowTo100M

1. **Download annotations**  
   - [HowToGround1M annotations](https://drive.google.com/drive/folders/1462NzzloYlTecJpn71jios4_cU6h4dQd?usp=sharing)  
   - [iGround annotations](https://drive.google.com/drive/folders/1tCGeiRCo5eYh6u8TL02OPOP42EiskNrp?usp=sharing)

2. **Preprocess annotations**
   - Run the following command to split the annotations into separate files per video:
     ```bash
     bash scripts/preprocess_howtoground_annot.py /path/to/{HowToGround1M,iGround}.pkl target_dir
     ```
3. **Download iGround videos** (for fine-tuning/evaluating on iGround)
   - Fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSeqYqoFludNXy-2iQHnmIJhN8FfB0ieqnz8GWWF6hqfAEZexw/viewform?usp=header) to obtain an access token
   - Run `$export TOKEN=YOUR_TOKEN`
   - Run the following script to download the iGround videos using the provided links
     ```bash
     bash scripts/download_datasets.sh iground /path/to/iground_videos_dir /path/to/iGround_annotations/*_raw.pkl
     ```

4. **Download HowToGround1M videos** (for pre-training on HowToGround1M)
   - Fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSeK7zHcN5udVrzjt5rs34cIbICuEXLJvxu8IxW-GfzObq3-qw/viewform?usp=dialog) to obtain an access token
   - Run `$export TOKEN=YOUR_TOKEN`
   - Run the following script to download the HowToGround1M videos using the provided links
     ```bash
     bash scripts/download_datasets.sh howtoground1m /path/to/howtoground1m_videos_dir /path/to/HowToGround1M_annotations/HowToGround1M_automatic_annotation_method.pkl
     ```
     
**Note**: The iGround annotations include both *processed* and *raw* versions (e.g., `iGround_train_set_processed.pkl` vs `iGround_train_set_raw.pkl`). The *processed* annotations were used to train GROVE. Processing merges multiple instances of the same object type **per frame** into a single annotation by taking the union of all bounding boxes for that instance. The *raw* annotations are unprocessed — the same object type may appear multiple times in a frame, each with its own distinct bounding box per instance.
<br>
### Access to the full HowTo100M dataset 
We also provide access to the original full HowTo100M dataset videos. To download:
 - Fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSdp_PlR2xnUM1RACyBdzbNt9mUnhARIfJs2JmTcLiHVRFjv7g/viewform?usp=dialog) to obtain an access token
 - Run `$export TOKEN=YOUR_TOKEN`
 - Run the following script to download the Howto100M videos using the provided links
   ```bash
   bash scripts/download_datasets.sh howto100m /path/to/howto100m_videos_dir
   ```

If you use HowTo100M in your research, please also cite:

```bibtex
@inproceedings{miech19howto100m,
   title={How{T}o100{M}: {L}earning a {T}ext-{V}ideo {E}mbedding by {W}atching {H}undred {M}illion {N}arrated {V}ideo {C}lips},
   author={Miech, Antoine and Zhukov, Dimitri and Alayrac, Jean-Baptiste and Tapaswi, Makarand and Laptev, Ivan and Sivic, Josef},
   booktitle={ICCV},
   year={2019},
}
```

## CAUTION: the links expire in 14 days!

---

### ActivityNet-Entities

1. **Download ActivityNet videos**  
   - From [Hugging Face](https://huggingface.co/datasets/YimuWang/ActivityNet)  

2. **Download annotations**  
   - [ActivityNet-Entities annotations](https://drive.google.com/drive/folders/1DXauLe7CbAjoiiNVIakqCLJkxh8kkpjI?usp=sharing)  

3. **Preprocess videos**  
   ```bash
   bash scripts/preprocess_anet_videos.sh input_dataset_dir preprocessed_dataset_dir
   ```

---

### VidSTG

1. **Download VidSTG videos**  
   - From [Hugging Face](https://huggingface.co/datasets/shangxd/vidor/tree/main)  

2. **Download annotations**  
   - [VidSTG annotations](https://drive.google.com/drive/folders/1uPP8admYuny0XYj0wpQrEg3YgJS3ZG75?usp=sharing)  
  
## Checkpoints

- Download GROVE pre-trained on HowToGround1M from [link](https://drive.google.com/file/d/1cH4HcDQoBvVpZw5TfHtWGmbIX82R6T-a/view?usp=sharing)
- Download GROVE fine-tuned on iGround from [link](https://drive.google.com/file/d/1gBanRUuz5CfNUWkRstFrQQpHpCzuhhnY/view?usp=sharing)
- Download SAM checkpoint from [link](https://drive.google.com/file/d/1Am_IUCaGsr0pMMb-WFJ3cv5KpJfSMTZR/view?usp=sharing)
- Run:
  ```bash
  mkdir checkpoints
  mv /path/to/checkpoints checkpoints/
  ```

## Training (using SLURM's *sbatch*)

- In `train_scripts/train_{howtoground,vidstg,anet}.sh`:
  1. (Optional) Modify the sbatch configuration based on your cluster's configuration, though it is suggested to use the provided ones
  2. Modify the path to the data and checkpoint
- Run:
  ```bash
  bash train_scripts/train_{howtoground,vidstg,anet}.sh
  ```

**Note**: `train_scripts/train_howtoground.sh` can be used for both HowToGround1M and iGround datasets.

## Inference & evaluation

Below, it is shown how to run inference & evaluation on iGround validation and test sets. Similarly, for the other datasets use the scripts found in `infer_eval_scripts/`

- For iGround validation set:
  ```bash
  bash infer_eval_scripts/infer_eval_iground.sh checkpoints/grove_ft_iground_ckpt.bin /path/to/save/token_embedings.pt /path/to/save/preds.pkl /path/to/iGround_val_set_raw.pkl /path/to/iground_videos_dir 0.5 /path/to/stanford-corenlp-full-2014-08-27
  ```
- For iGround test set:
  ```bash
  bash infer_eval_scripts/infer_eval_iground.sh checkpoints/grove_ft_iground_ckpt.bin /path/to/save/token_embedings.pt /path/to/save/preds.pkl /path/to/iGround_test_set_raw.pkl /path/to/iground_videos_dir 0.5 /path/to/stanford-corenlp-full-2014-08-27
  ```
**Note**: By downloading Stanford CoreNLP from the links provided in the installation instructions, you will get a directory `stanford-corenlp-full-2014-08-27` which contains Stanford CoreNLP 3.4.1 (used above for evaluation in iGround) and a directory `stanford-corenlp-4.5.7` which contains Stanford CoreNLP 4.5.7 (used for evaluation in ActivityNet-Entities).
