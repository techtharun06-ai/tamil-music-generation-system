# 🎵 Tamil Background Music Generation System - Model 4

**Production-Grade Instrumental Accompaniment Generator for Tamil Music**

[![GitHub](https://img.shields.io/badge/GitHub-techtharun06--ai-blue)](https://github.com/techtharun06-ai/tamil-music-generation-system)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/TPU-Kaggle-orange)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 System Overview

Model 4 generates complete **instrumental accompaniment** that:

✅ **Matches original vocal characteristics** (F0-based pitch alignment)  
✅ **Maintains harmonic compatibility** with detected musical key  
✅ **Preserves rhythmic consistency** (tala/rhythm-aware generation)  
✅ **Adheres to Tamil genre conventions** (Hip-Hop, Melody, Kuthu, Carnatic)  
✅ **Produces studio-quality stereo audio** (44.1 kHz, -18 LUFS loudness)  

**Output Format:** 44.1 kHz Stereo WAV + MP3

---

## 📊 Dataset

**Total: 2800 Tamil Songs** (Both Google Drive Folders Combined)

### Genre Distribution:
- **Hip-Hop** (~700 songs) - Modern Tamil film music, electronic beats
- **Melody** (~700 songs) - Melodic focus, mixed instrumentation
- **Kuthu** (~700 songs) - Folk/Devotional, traditional rhythms
- **Carnatic** (~700 songs) - Classical Tamil music, microtonal

### Data Format:
- **Audio:** MP3 format (source with original vocals)
- **Lyrics:** LRC synchronized format (timestamp + Tamil text)
- **Source:** Google Drive (2 folders)
- **Processing:** Demucs vocal separation → feature extraction → training

---

## 🏗️ Architecture

### Two-Stage Pipeline

#### **Stage A: Arrangement Generator** 🎼
- **Model:** Transformer-XL (100M parameters, TPU-optimized)
- **Input:** Vocal melody tokens + genre embedding + tala + BPM + mood
- **Output:** Multi-track MIDI arrangement (drums, bass, harmony, pads)
- **Token Format:** REMI+ (note on/off, velocity, time shift, program change)
- **Context Length:** 4096 tokens
- **Training:** 200K steps on 2800 songs

#### **Stage B: Audio Renderer** 🔊
- **Model:** AudioLDM-style Latent Diffusion (30M parameters)
- **Input:** MIDI arrangement + genre embedding + mood embedding
- **Output:** 44.1 kHz stereo instrumental audio (WAV)
- **Diffusion Sampler:** DDIM (50 steps for inference)
- **VAE:** Mel-spectrogram compression (8x time downsampling)
- **Training:** 500K+ steps on 2800 songs

---

## 🎼 Model 3 Integration Strategy

**Current Status:** Training with original vocals as reference signal

### Why This Approach?
1. ✅ **Immediate training** on 2800 real songs with authentic vocal characteristics
2. ✅ **F0 extraction** from original audio using CREPE pitch estimator
3. ✅ **Vocal onset detection** for beat-level synchronization
4. ✅ **Frequency-aware harmonization** (avoids vocal frequency range 80-400 Hz)
5. ✅ **Dynamic ducking** learned during training (-6 to -10 dB during vocals)

### When Model 3 Ready
- ✅ **No retraining required** - simply swap vocal reference
- ✅ **Perfect synchronization maintained** (system already vocal-aware)
- ✅ **Drop-in replacement** architecture

---

## 📁 Project Structure

```
tamil-music-generation-system/
├── README.md
├── requirements.txt
├── setup.sh
├── LICENSE
├── .gitignore
│
├── config/
│   ├── stage_a_config.yaml
│   ├── stage_b_config.yaml
│   └── genre_profiles.yaml
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── gdrive_downloader.py
│   │   ├── audio_processor.py
│   │   ├── vocal_separator.py
│   │   ├── feature_extractor.py
│   │   ├── lrc_parser.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arrangement_generator.py
│   │   ├── audio_renderer.py
│   │   ├── tokenizer.py
│   │   └── diffusion.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── stage_a_trainer.py
│   │   ├── stage_b_trainer.py
│   │   └── utils.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── arrangement_inference.py
│   │   ├── audio_rendering.py
│   │   └── post_processing.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── config_loader.py
│
├── scripts/
│   ├── download_dataset.py
│   ├── preprocess_data.py
│   ├── train_stage_a.py
│   ├── train_stage_b.py
│   ├── inference.py
│   └── evaluate.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_stage_a_training.ipynb
│   ├── 03_stage_b_training.ipynb
│   └── kaggle_tpu_training.ipynb
│
└── outputs/
    ├── checkpoints/
    ├── generated_audio/
    └── logs/
```

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/techtharun06-ai/tamil-music-generation-system.git
cd tamil-music-generation-system
bash setup.sh
```

### 2️⃣ Download Dataset
```bash
python scripts/download_dataset.py \
  --folder-ids "1I5F4uzlfb5qM01jLZzeY5GB6cGIlXIri,1ioIaQD1ysYKpOuWh6wtRjGbe32B2SV4R" \
  --lyrics-ids "1SE1-OJ4NyVnkQ_6urgeotUjrn99SpSrV,1BPxL0RGRSe5pHPrrDdVBny2ZoDWAb8pE" \
  --folder-names "Folder1_Audio,Folder2_Audio" \
  --lyrics-names "Folder1_Lyrics,Folder2_Lyrics" \
  --output-dir ./data/raw \
  --organize \
  --save-metadata
```

### 3️⃣ Preprocess Data
```bash
python scripts/preprocess_data.py \
  --input-dir ./data/raw \
  --output-dir ./data/processed \
  --num-workers 8
```

### 4️⃣ Train Stage A
```bash
python scripts/train_stage_a.py \
  --data-dir ./data/processed \
  --batch-size 32 \
  --epochs 50 \
  --device tpu
```

### 5️⃣ Train Stage B
```bash
python scripts/train_stage_b.py \
  --data-dir ./data/processed \
  --batch-size 8 \
  --epochs 100 \
  --device tpu
```

### 6️⃣ Generate Accompaniment
```bash
python scripts/inference.py \
  --song-id 42 \
  --stage-a-checkpoint ./outputs/checkpoints/stage_a/best.pt \
  --stage-b-checkpoint ./outputs/checkpoints/stage_b/best.pt \
  --output-dir ./outputs/generated_audio
```

---

## 🎯 Supported Tamil Genres

| Genre | Characteristics | Instruments |
|-------|-----------------|-------------|
| **Hip-Hop** | Modern, electronic beats | Synth, Drums, Bass |
| **Melody** | Melodic focus, mixed | Piano, Violin, Strings |
| **Kuthu** | Folk, traditional rhythms | Thavil, Nadaswaram, Percussion |
| **Carnatic** | Classical, microtonal, complex talas | Veena, Mridangam, Violin |

---

## 📊 Training Configuration

### Stage A
```yaml
Batch Size:       32 (TPU-optimized)
Learning Rate:    5e-4
Training Steps:   200,000
Mixed Precision:  bfloat16
```

### Stage B
```yaml
Batch Size:       8
Learning Rate:    1e-4
Training Steps:   500,000+
Mixed Precision:  bfloat16
Diffusion Steps:  50 (DDIM)
```

---

## 📈 Evaluation Metrics

- **Fréchet Audio Distance (FAD)** - Audio quality
- **Beat Alignment Accuracy** - Rhythm synchronization
- **Chroma Consistency** - Harmonic compatibility
- **Genre Classification** - Cultural authenticity
- **Human Evaluation** - Musicality & naturalness

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙋 Support

GitHub: [@techtharun06-ai](https://github.com/techtharun06-ai)  
Issues: [Report bugs](https://github.com/techtharun06-ai/tamil-music-generation-system/issues)

---

**Status:** 🔴 Active Development  
**Dataset:** 2800 Tamil Songs  
**Last Updated:** June 9, 2026
