<p align="center">
  <img src="assets/icon.png" alt="Dolphin Logo" width="150"/>
</p>
<h3 align="center">Dolphin: Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention</h3>
<p align="center">
  <strong>Kai Li*, Kejun Gao*, Xiaolin Hu </strong><br>
  <strong>Tsinghua University</strong>
</p>

<p align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.Dolphin" alt="访客统计" />
  <img src="https://img.shields.io/github/stars/JusperLee/Dolphin?style=social" alt="GitHub stars" />
  <img alt="Source code license" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
  <a href="https://openreview.net/forum?id=LaIkPfPu9K" target="_blank" rel="noreferrer noopener">
    <img alt="ICLR 2026 OpenReview" src="https://img.shields.io/badge/ICLR%202026-OpenReview-8c1b13.svg?logo=openreview&logoColor=white" />
  </a>
  <a href="https://huggingface.co/JusperLee/Dolphin" target="_blank" rel="noreferrer noopener">
    <img alt="Hugging Face Models" src="https://img.shields.io/badge/Hugging%20Face-Models-ff9d2c?logo=huggingface&logoColor=white" />
  </a>
  <a href="https://huggingface.co/spaces/JusperLee/Dolphin" target="_blank" rel="noreferrer noopener">
    <img alt="Hugging Face Spaces" src="https://img.shields.io/badge/%20Hugging%20Face-Space-yellow.svg?logo=huggingface&logoColor=white" />
  </a>
  <a href="https://cslikai.cn/Dolphin" target="_blank" rel="noreferrer noopener">
    <img alt="Demo Page" src="https://img.shields.io/badge/Demo-Page-blue.svg" />
  </a>
</p>

<p align="center">

> Dolphin is an efficient audio-visual speech separation framework that leverages discrete lip semantics and global–local attention to achieve state-of-the-art performance with significantly reduced computational complexity.

## 🎯 Highlights

- **Balanced Quality & Efficiency**: Single-pass separator achieves state-of-the-art AVSS performance without iterative refinement.
- **DP-LipCoder**: Dual-path, vector-quantized video encoder produces discrete audio-aligned semantic tokens while staying lightweight.
- **Global–Local Attention**: TDANet-based separator augments each layer with coarse global self-attention and heat diffusion local attention.
- **Edge-Friendly Deployment**: Delivers >50% parameter reduction, >2.4× lower MACs, and >6× faster GPU inference versus IIANet.

## 💥 News

- **[2026-04-14]** Added HuggingFace Hub evaluation support in `eval.py` — load pretrained models directly with `--use_hf` flag. 🤗
- **[2026-04-09]** Released `dolphin_light.py` — a memory-efficient variant that replaces the explicit cosine matrix in Heat1D with FFT-based DCT/IDCT, reducing memory from O(T²) to O(T log T) for long sequences.  Thanks to @Shuhan Zhang, and @Wenxuan Wu for reporting this issue! 🔧
- **[2026-03-10]** Added video encoder pretraining and audio training code, together with updated training instructions in README. 🚀
- **[2026-01-26]** Dolphin was accepted to ICLR 2026. 🎉
- **[2025-09-28]** Code and pre-trained models are released! 📦

## 📜 Abstract

Audio-visual speech separation (AVSS) methods leverage visual cues to extract target speech in noisy acoustic environments, but most existing systems remain computationally heavy. Dolphin tackles this tension by combining a lightweight, dual-path video encoder with a single-pass global–local collaborative separator. The video pathway, DP-LipCoder, maps lip movements into discrete semantic tokens that remain tightly aligned with audio through vector quantization and distillation from AV-HuBERT. The audio separator builds upon TDANet and injects global–local attention (GLA) blocks—coarse-grained self-attention for long-range context and heat diffusion attention for denoising fine details. Across three public AVSS benchmarks, Dolphin not only outperforms the state-of-the-art IIANet on separation metrics but also delivers over 50% fewer parameters, more than 2.4× lower MACs, and over 6× faster GPU inference, making it practical for edge deployment.

## 🌍 Motivation

In real-world environments, target speech is often masked by background noise and interfering speakers. This phenomenon reflects the classic “cocktail party effect,” where listeners selectively attend to a single speaker within a noisy scene (Cherry, 1953). These challenges have spurred extensive research on speech separation.

Audio-only approaches tend to struggle in complex acoustic conditions, while the integration of synchronous visual cues offers greater robustness. Recent deep learning-based AVSS systems achieve strong performance, yet many rely on computationally intensive separators or heavy iterative refinement, limiting their practicality.

Beyond the separator itself, AVSS models frequently inherit high computational cost from their video encoders. Large-scale lip-reading backbones provide rich semantic alignment but bring prohibitive parameter counts. Compressing them often erodes lip semantics, whereas designing new lightweight encoders from scratch risks losing semantic fidelity and degrading separation quality. Building a video encoder that balances compactness with semantic alignment therefore remains a central challenge for AVSS.

## 🧠 Method Overview

To address these limitations, Dolphin introduces a novel AVSS pipeline centered on two components:

- **DP-LipCoder**: A dual-path, vector-quantized video encoder that separates compressed visual structure from audio-aligned semantics. By combining vector quantization with knowledge distillation from AV-HuBERT, it converts continuous lip motion into discrete semantic tokens without sacrificing representational capacity.
- **Single-Pass GLA Separator**: A lightweight TDANet-based audio separator that removes the need for iterative refinement. Each layer hosts a global–local attention block: coarse-grained self-attention captures long-range dependencies at low resolution, while heat diffusion attention smooths features across channels to suppress noise and retain detail.

Together, these components strike a balance between separation quality and computational efficiency, enabling deployment in resource-constrained scenarios.

## 🧪 Experimental Highlights

We evaluate Dolphin on LRS2, LRS3, and VoxCeleb2. Compared with the state-of-the-art IIANet, Dolphin achieves higher scores across all separation metrics while dramatically reducing resource consumption:

- **Parameters**: >50% reduction
- **Computation**: >2.4× decrease in MACs
- **Inference**: >6× speedup on GPU

These results demonstrate that Dolphin provides competitive AVSS quality on edge hardware without heavy iterative processing.

## 🏗️ Architecture

![Dolphin Architecture](assets/overall-pipeline.png)

> The overall architecture of Dolphin.

### Video Encoder

![Dolphin Architecture](assets/video-ae.png)

> The video encoder of Dolphin.

### Dolphin Model Overview

![Dolphin Architecture](assets/separator.png)

> The overall architecture of Dolphin's separator.

### Key Components

![Dolphin Architecture](assets/ga-msa.png)

1. **Global Attention (GA) Block**
   - Applies coarse-grained self-attention to capture long-range structure
   - Operates at low spatial resolution for efficiency
   - Enhances robustness to complex acoustic mixtures

2. **Local Attention (LA) Block**
   - Uses heat diffusion attention to smooth features across channels
   - Suppresses background noise while preserving details
   - Complements GA to balance global context and local fidelity

## 📊 Results

### Performance Comparison

Performance metrics on three public AVSS benchmark datasets. Bold indicates best performance.

![Results Table](assets/results.png)

### Efficiency Analysis

![Efficiency Comparison](assets/efficiency_comparison.png)

Dolphin achieves:
- ✅ **>50%** parameter reduction
- ✅ **2.4×** lower computational cost (MACs)
- ✅ **6×** faster GPU inference speed
- ✅ Superior separation quality across all metrics

## 📦 Installation

```bash
git clone https://github.com/JusperLee/Dolphin.git
cd Dolphin
pip install torch torchvision
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.5.0
- CUDA >= 12.4
- Other dependencies in requirements.txt

## 🔍 Inference with Pre-trained Model

```python
# Single audio-visual separation
python inference.py \
    --input /path/to/video.mp4 \
    --output /path/to/output/directory \
    --speakers 2 \
    --detect-every-n 8 \
    --face-scale 1.5 \
    --cuda-device 0 \
    --config checkpoints/vox2/conf.yml
```

## 📁 Model Zoo

| Model | Training Data | SI-SNRi | PESQ | Download |
|-------|--------------|---------|------|----------|
| Dolphin | VoxCeleb2 | 14.6 dB | 3.17 | [Link](https://huggingface.co/JusperLee/Dolphin) |

## 🚀 Training Pipeline

### Step 1. Video Encoder Pretraining (Placeholder)

1) You can first prepare the video-only file list:

```bash
python DataPreProcess/process_videoonly.py \
  --in_dirs /path/to/lrs2/mouths /path/to/lrs3/mouths /path/to/vox2/mouths \
  --out_dir DataPreProcess/video_pretrain.txt

python DataPreProcess/split_dataset.py \
  --input DataPreProcess/video_pretrain.txt \
  --output_dir DataPreProcess/video_pretrain \
  --train_ratio 0.8 \
  --seed 42

```

2) Install AV-HuBERT:

```bash
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert
git submodule init
git submodule update
```

Lastly, install Fairseq and the other packages:

```bash
pip install -r requirements.txt
cd fairseq
pip install --editable ./
```

3) Extract AV-HuBERT mouth features:

```bash
cd ../../
wget -O videoencoder_pretrain/large_vox_433h.pt \
  https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/large_vox_433h.pt

torchrun --nproc_per_node=8 videoencoder_pretrain/extract_avhubert_mouth_features.py \
  --in_dirs /path/to/lrs2/mouths /path/to/lrs3/mouths /path/to/vox2/mouths \
  --ckpt_path videoencoder_pretrain/large_vox_433h.pt \
  --output_dir videoencoder_pretrain \
  --output_prefix videodata_3dataset \
  --merged_output_name videodata_large_3datasets.pth
```

4) Pretrain the video encoder:
```bash
python videoencoder_pretrain/pretrain.py --conf_dir=configs/videoencoder_pretrain.yml
```

### Step 2. Separator Training (after video encoder pretraining)

> ❗ Reminder: You can skip Step 1 and directly use the provided pretrained video encoder for separator training. However, if you are using a third-party dataset other than LRS2, LRS3, or Vox2, we recommend rerunning Step 1 to pretrain the video encoder on your own data.

1) Prepare your dataset JSONs (e.g., `mix.json`, `s1.json`, `s2.json`) under:

```bash
python DataPreProcess/process_lrs23.py \
  --in_audio_dir lrs3_rebuild/audio/wav16k/min \
  --in_mouth_dir lrs3_rebuild/mouths \
  --out_dir DataPreProcess/LRS3

python DataPreProcess/process_vox2.py \
  --in_audio_dir vox2/audio_10w/wav16k/min \
  --in_mouth_dir vox2/mouths \
  --out_dir DataPreProcess/vox2
```

2) Check key fields in `configs/dolphin.yml`:
- Basic training fields:
```yaml
datamodule:
  data_name: AVSpeechDataModule

audionet:
  audionet_name: Dolphin
  audionet_config:
    is_train: true

training:
  system: AudioVisualLightningModuleAE

loss:
  train:
    loss_func: MultiDecoder_PITLossWrapper
```
- Configure video encoder weights:
  Keep the following if you load from Hugging Face:
```yaml
audionet:
  audionet_config:
    video_encoder_pretrained_hf:
      source: "huggingface"
      model_id: "JusperLee/Dolphin"
      revision: "main"
      filename: "model.safetensors"
      strict: false
```
  Change it to the following if you use a local checkpoint such as `Experiments/checkpoint/videoencoder-pretrain/best_model.pth`:
```yaml
audionet:
  audionet_config:
    video_encoder_pretrained_hf:
      source: "local"
      path: "Experiments/checkpoint/videoencoder-pretrain/best_model.pth"
      strict: false
```
  `source` supports `huggingface` / `hf` / `hub` and `local` / `pth`.
  `path` can be either a relative path from the project root or an absolute path.
  The local checkpoint must contain `video_encoder.*` weights.
- Configure experiment logging if needed:
```yaml
logging:
  use_swanlab: true
  swanlab:
    project: "test-dolphin"
    workspace: "your_swanlab_workspace"
    offline: false
```
  `logging.use_swanlab: true` enables SwanLab; if set to `false`, training falls back to TensorBoard automatically.
  `logging.swanlab.project` is the project name shown in SwanLab.
  `logging.swanlab.workspace` is your SwanLab workspace or username.
  `logging.swanlab.offline: true` means offline logging only; `false` means upload online.

3) Start training:
```bash
python train.py --conf_dir=configs/dolphin.yml
```

During startup, you should see logs like:
- `[Dolphin] video_encoder loaded: ...`
- `[Dolphin] video_encoder frozen: ... trainable_params=0`

### Step 3. Evaluation

After training completes, run evaluation on the test set:

```bash
python eval.py --conf_dir=Experiments/checkpoint/<exp_name>/conf.yml
```

You can also evaluate directly using the pretrained model from 🤗 HuggingFace Hub:

```bash
python eval.py \
  --conf_dir=configs/dolphin.yml \
  --use_hf \
  --hf_model_id JusperLee/Dolphin \
  --test_dir /path/to/your/test/set
```

| Argument | Description |
|----------|-------------|
| `--conf_dir` | Path to config YAML (training `conf.yml` or `configs/dolphin.yml`) |
| `--use_hf` | Load model from HuggingFace Hub instead of local checkpoint |
| `--hf_model_id` | HuggingFace model ID (default: `JusperLee/Dolphin`) |
| `--test_dir` | Override test set directory from config |

The script computes SI-SNR, SI-SNRi, SDR, SDRi, PESQ, and STOI, saves per-utterance results to `results/metrics.csv`, and displays live metrics every 50 samples in the progress bar.

## 📖 Citation

If you find Dolphin useful in your research, please cite:

```bibtex
@inproceedings{li2026efficientaudiovisualspeechseparation,
      title={Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention},
      author={Kai Li and Kejun Gao and Xiaolin Hu},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2026}
}
```

## 🤝 Acknowledgments

We thank the authors of [IIANet](https://github.com/JusperLee/IIANet) and [SepReformer](https://github.com/dmlguq456/SepReformer) for providing parts of the code used in this project.

## 📧 Contact

For questions and feedback, please open an issue on GitHub or contact us at: [tsinghua.kaili@gmail.com](tsinghua.kaili@gmail.com)

## 📄 Licensing and Model Provenance

Except where a file-level notice states otherwise, Dolphin-owned source code
in this repository is licensed under the
[Apache License 2.0](LICENSE). Portions adapted from IIANet and SepReformer
are also distributed under Apache-2.0. See [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution details.

The source-code license does **not** by itself grant rights to the released
model weights, training datasets, source recordings, or other third-party
materials. The currently released checkpoint contains a video encoder trained
with feature targets produced by AV-HuBERT. No separate permission from Meta
is included with this release.

Pending written clarification of the relevant upstream terms and completion
of our data-provenance review, we do not represent the current checkpoint as
cleared for commercial or production use. Please read
[MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) before using the model artifacts.

<p align="center">
  Made with stars ⭐️ for efficient audio-visual speech separation
</p>
