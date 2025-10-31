# GStex-CTRL: Text-Driven Controllable 3D Editing with GStex

**Authors:** Qingyang Bao, Victor Rong, David Lindell  
**Conference:** Project Report based on WACV 2025 work on [GStex](https://arxiv.org/abs/2409.12954)

---

## Overview

**GStex-CTRL** introduces a **text-driven controllable 3D appearance editing pipeline** that combines the strengths of **GStex** (per-primitive texturing for 2D Gaussian Splatting) and **ControlNet-based diffusion editing**.  
Our goal is to **decouple geometry and appearance** in 3D Gaussian Splatting (3DGS) and enable fine-grained **text-conditioned appearance edits** while maintaining strong **multi-view consistency**.

<p align="center">
  <img src="assets/overview.png" width="700"/>
</p>

---

## Key Features

- **🔹 Text-Driven Editing:** Modify 3D scenes using natural language prompts.  
- **🔹 More detailed Appearance:** Achieved via GStex’s per-Gaussian texture maps.  
- **🔹 ControlNet Depth Conditioning:** Ensures spatially consistent editing across multiple views.  
- **🔹 Compatible with Nerfstudio:** Built on top of the modular [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework.  
- **🔹 Fast and High-Quality Rendering:** Inherits GStex’s efficiency and visual fidelity.  

---

## Method Overview

The GStex-CTRL pipeline extends **GaussCtrl (ECCV 2024)** by replacing its 3D Gaussian Splatting representation with **GStex**.  
The workflow proceeds as follows:

1. **Reconstruct Scene** using **GStex** and COLMAP.  
2. **Render RGB + Depth Images** from the GStex scene.  
3. **Apply ControlNet Editing** conditioned on depth and text prompts.  
4. **Optimize the GStex Model** using the edited renders to produce the final 3D scene.  

<p align="center">
  <img src="assets/pipeline.png" width="700"/>
</p>

---

## Experiments

We evaluate GStex-CTRL across several benchmark scenes:

- **Datasets:** InstructNeRF2NeRF, Mip-NeRF360, BlendedMVS  
- **Evaluation Metric:** CLIP Text-Image Directional Similarity  

## 📚 Citation

If you find this project helpful, please cite the following:

> **Bao, Qingyang; Rong, Victor; Lindell, David.**  
> *Text-Driven Controllable 3D Editing with GStex.*  
> 2025.  

**BibTeX:**
```bibtex
@article{bao2025gstexctrl,
  title={Text-Driven Controllable 3D Editing with GStex},
  author={Bao, Qingyang and Rong, Victor and Lindell, David},
  year={2025}
