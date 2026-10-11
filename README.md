# FaceSwap: Deep Learning-Based Face Swapping Pipeline

A PyTorch-powered deep learning framework for high-fidelity, automated face swapping in video content. The system uses a shared-encoder dual-decoder autoencoder architecture combined with adversarial training, MTCNN face detection, and Delaunay triangulation warping for seamless geometric blending.

---

## Table of Contents
- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [1. Face Extraction](#phase-1-face-detection--extraction)
  - [2. Model Training](#phase-2-autoencoder-training)
  - [3. Video Face Swapping](#phase-3-video-reconstruction--inference)
  - [Jupyter Notebook Usage](#jupyter-notebook)
- [CLI Reference](#cli-reference)
- [Performance & Hardware Guidelines](#performance--hardware-guidelines)
- [Troubleshooting](#troubleshooting)
- [Documentation Index](#documentation-index)

---

## Project Overview

FaceSwap allows bidirectional face replacement between two people across video streams:
- **Shared Latent Representation**: A single encoder maps faces of both Person A and Person B into a shared latent space capturing universal facial geometry, pose, and expression.
- **Identity-Specific Decoders**: Two distinct decoders learn person-specific facial details, textures, and features. Swapping the decoder during inference replaces Person A's identity while maintaining their original expression and head pose.
- **Geometric Landmark Blending**: Employs MTCNN and dlib 68-point facial landmarks combined with Delaunay triangulation to align and seamlessly warp generated faces onto destination video frames.
- **Composite Loss Function**: Combines $L_1$ reconstruction loss, Structural Similarity Index Measure (SSIM) loss for perceptual realism, and optional GAN discriminator loss.

---

## How It Works

```
      Input Frame (Person A)
                │
                ▼
     Face Extraction (MTCNN)
                │
                ▼
    Shared Encoder (Latent Code)
           ╱            ╲
          ▼              ▼
     Decoder A      Decoder B (Swap)
   (Reconstruct A)  (Person B with A's pose)
                         │
                         ▼
             Delaunay Triangulation
                & Seamless Blending
                         │
                         ▼
             Swapped Output Frame
```

During training:
- **Image A** $\rightarrow$ **Encoder** $\rightarrow$ **Decoder A** $\rightarrow$ **Reconstruction Loss ($A \leftrightarrow \hat{A}$)**
- **Image B** $\rightarrow$ **Encoder** $\rightarrow$ **Decoder B** $\rightarrow$ **Reconstruction Loss ($B \leftrightarrow \hat{B}$)**

During inference:
- **Video Frame A** $\rightarrow$ **Encoder** $\rightarrow$ **Decoder B** $\rightarrow$ **Face with Person B's features and Person A's expression**

---

## System Architecture

```mermaid
flowchart TD
    subgraph DataPrep [Phase 1: Extraction & Alignment]
        V1[Source Video A] --> FD1[face_detect.py / MTCNN]
        V2[Target Video B] --> FD2[face_detect.py / MTCNN]
        FD1 --> D1[Aligned Face Set A (128x128)]
        FD2 --> D2[Aligned Face Set B (128x128)]
    end

    subgraph Training [Phase 2: Autoencoder Training]
        D1 --> E[Shared Encoder]
        D2 --> E
        E --> DA[Decoder A]
        E --> DB[Decoder B]
        DA --> L1[L1 + SSIM Loss]
        DB --> L2[L1 + SSIM Loss]
        L1 & L2 --> W[saved_models/faceswap_model.pt]
    end

    subgraph Inference [Phase 3: Video Inference & Blending]
        OrigVid[Original Video A] --> VW[video_writer.py]
        W --> VW
        VW --> Ext[Frame Extraction & MTCNN]
        Ext --> EInf[Shared Encoder]
        EInf --> DBInf[Decoder B]
        DBInf --> DT[faceswap.py: Delaunay Triangulation & Seamless Clone]
        DT --> OutVid[Output Video: Person A with B's Face]
    end
```

---

## Directory Structure

```
faceswap1/
├── DeepFakeTorch/
│   ├── face_detect.py                 # Face detection & extraction CLI (MTCNN)
│   ├── faceswap.py                    # Delaunay triangulation & warp blending
│   ├── img_rotate.py                  # Face rotation & alignment utilities
│   ├── Iter.py                        # Dataset iterator & batch handling
│   ├── model.py                       # Autoencoder & Discriminator architectures
│   ├── SSIM.py                        # Structural Similarity (SSIM) loss module
│   ├── train.py                       # Training loop with multi-component loss
│   ├── video_writer.py                # Frame-by-frame video inference & writer
│   ├── writes_images.py               # Image writing & debugging utilities
│   ├── mmod_human_face_detector.dat   # dlib face detector model weights
│   ├── requirements.txt               # Python package dependencies
│   ├── LICENSE                        # Project license
│   └── images/                        # Sample result animations
├── FaceSwap.ipynb                     # End-to-end interactive Jupyter notebook
├── ARCHITECTURE.md                    # Deep dive into network layers & dataflow
├── WORKFLOW.md                        # Step-by-step pipeline operational manual
├── SETUP_AND_USAGE.md                 # Detailed setup & command guide
├── PROJECT_SUMMARY.md                 # Executive summary & interview talking points
├── QUICK_REFERENCE.md                 # Fast reference cheatsheet
├── IMAGE_RECONSTRUCTION_TECHNIQUES.md # Mathematical breakdown of loss & warping
├── RECONSTRUCTION_VISUAL_GUIDE.md     # Visual illustrations of reconstruction
├── INTERVIEW_REFERENCE.md             # Q&A guide for portfolio reviews
└── README.md                          # Master documentation (this file)
```

---

## Prerequisites & Installation

### 1. System Requirements
- **OS**: Windows 10/11 or Linux
- **Python**: 3.7 - 3.8
- **NVIDIA GPU**: Recommended (GTX 1060 or better; 6GB+ VRAM recommended for fast training)
- **CMake**: Required for building `dlib`

### 2. Environment Setup

```bash
# Create and activate a clean conda environment
conda create -n faceswap python=3.8 -y
conda activate faceswap

# Install CMake if building dlib from source
conda install -c conda-forge cmake -y
```

### 3. Install Dependencies

Navigate to `DeepFakeTorch/` and install requirements:

```bash
cd DeepFakeTorch
pip install -r requirements.txt
```

#### Core Dependencies:
- `torch` & `torchvision` (CUDA enabled)
- `facenet_pytorch` (MTCNN detection)
- `opencv-python` (Image and video I/O)
- `dlib` (Landmark extraction)
- `numpy`, `scikit-image`, `Pillow`

### 4. Facial Landmark Predictor (dlib)
Download dlib's 68-point face landmark predictor and place it inside `DeepFakeTorch/`:
- [`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

---

## Step-by-Step Workflow

### Phase 1: Face Detection & Extraction
Extract and align face crops (128x128 resolution) from source videos:

```bash
cd DeepFakeTorch

# Extract faces for Person A
python face_detect.py -video_src path/to/person_a.mp4 -out_name faces_a -align_faces 1

# Extract faces for Person B
python face_detect.py -video_src path/to/person_b.mp4 -out_name faces_b -align_faces 1
```

*Output folders `faces_a/` and `faces_b/` will contain normalized, cropped faces.*

---

### Phase 2: Autoencoder Training
Train the shared encoder and dual decoders:

```bash
python train.py -face_a_dir faces_a \
               -face_b_dir faces_b \
               -batch_size 1 \
               -n_steps 10000 \
               -save_iter 1000 \
               -model_name faceswap_model
```

#### Training Recommendations:
| Target Quality | Iterations | Approx. Time (RTX 2080) | Notes |
|---|---|---|---|
| Quick Prototype | 5,000 | ~45 min | Fast convergence check |
| Standard Quality | 10,000 - 20,000 | 1.5 - 3 hours | Good expressions & textures |
| Production / HD | 50,000+ | 5 - 10 hours | Enable `-discriminator True` |

Checkpoints are saved in `saved_models/` every `-save_iter` steps.

---

### Phase 3: Video Reconstruction & Inference
Apply the trained network to produce the swapped video:

```bash
# Swap Person A's face with Person B's face:
python video_writer.py -original_video path/to/person_a.mp4 \
                       -model_location saved_models/faceswap_model.pt \
                       -decoder b \
                       -out_name output_a_with_b_face

# (Optional) Reverse swap: Person B's face with Person A's face:
python video_writer.py -original_video path/to/person_b.mp4 \
                       -model_location saved_models/faceswap_model.pt \
                       -decoder a \
                       -out_name output_b_with_a_face
```

---

### Jupyter Notebook
For interactive experimentation, open and run [`FaceSwap.ipynb`](FaceSwap.ipynb). It includes:
- Step-by-step visual inspection of detected faces.
- In-notebook training loop with live loss plots.
- Visual validation of reconstructions prior to full video rendering.

---

## CLI Reference

### `face_detect.py`
| Argument | Type | Default | Description |
|---|---|---|---|
| `-video_src` | `str` | Required | Path to input video file |
| `-out_name` | `str` | `'faces'` | Destination folder for extracted frames |
| `-align_faces` | `int` | `1` | `1` to align eyes horizontally; `0` for raw crops |
| `-margin` | `int` | `5` | Bounding box margin around detected face (px) |

### `train.py`
| Argument | Type | Default | Description |
|---|---|---|---|
| `-face_a_dir` | `str` | Required | Directory with Person A face images |
| `-face_b_dir` | `str` | Required | Directory with Person B face images |
| `-n_steps` | `int` | `10000` | Total training iterations |
| `-batch_size` | `int` | `1` | Batch size per training step |
| `-save_iter` | `int` | `1000` | Checkpoint frequency (in iterations) |
| `-model_name` | `str` | `'model'` | Saved model base filename |
| `-discriminator` | `bool` | `False` | Toggle adversarial GAN loss |

### `video_writer.py`
| Argument | Type | Default | Description |
|---|---|---|---|
| `-original_video` | `str` | Required | Input video to process |
| `-model_location` | `str` | Required | Path to trained `.pt` weights |
| `-decoder` | `str` | `'b'` | Decoder to apply (`'a'` or `'b'`) |
| `-out_name` | `str` | `'swapped'` | Output video filename (without extension) |

---

## Performance & Hardware Guidelines

- **GPU Acceleration**: CUDA is automatically selected when available. Training on CPU is possible but significantly slower.
- **VRAM Optimization**:
  - Keep `-batch_size 1` for 4GB - 8GB GPUs.
  - If CUDA Out of Memory occurs, disable `-discriminator` or resize input training crops to 96x96 in `train.py`.
- **Dataset Size**: 300 to 1,500 diverse face crops per person (varying head poses, angles, and lighting) yields optimal fidelity without overfitting.

---

## Troubleshooting

- **No face detected in video**: Ensure good lighting and unobstructed faces in the source footage. Adjust `-margin` in `face_detect.py`.
- **CUDA Out of Memory**: Lower batch size, free other GPU processes, or run with `-discriminator False`.
- **Edge Seam / Blending Artifacts**: Ensure `shape_predictor_68_face_landmarks.dat` is downloaded and present; Delaunay triangulation relies on accurate landmarks for boundary feathering.
- **Output Video Format Issues**: Re-encode with FFmpeg if the output container is unsupported:
  ```bash
  ffmpeg -i output_a_with_b_face.mp4 -c:v libx264 -pix_fmt yuv420p final_swapped.mp4
  ```

---

## Documentation Index

For deeper technical explorations, consult the companion guides:
- [ARCHITECTURE.md](ARCHITECTURE.md) — Neural network topology, latent space, and layer dimensions.
- [WORKFLOW.md](WORKFLOW.md) — Detailed operational flow and pipeline milestones.
- [SETUP_AND_USAGE.md](SETUP_AND_USAGE.md) — Comprehensive installation and troubleshooting manual.
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) — Key conceptual takeaways and interview prep.
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) — Command cheat sheet.
- [IMAGE_RECONSTRUCTION_TECHNIQUES.md](IMAGE_RECONSTRUCTION_TECHNIQUES.md) — Mathematical formulations of SSIM and Delaunay warping.
