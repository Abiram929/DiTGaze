# DiTGaze: Geometry-Conditioned Diffusion Augmentation for Few-Shot Gaze Estimation

> [Paper](https://arxiv.org/abs/2511.11231) | [Code](https://github.com/Abiram929/DiTGaze)
>
> **Abiram Panchalingam**, **Indu P. Bodala**, **Stuart E. Middleton**  
> University of Southampton  
> Accepted at the **IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2027**

> [!IMPORTANT]
> **Release status:** This repository is currently undergoing final preparation for the official DiTGaze research-code release. The complete tested implementation, checkpoints and evaluation instructions will be added shortly. The current version should not yet be treated as the reproducibility release.

DiTGaze is a calibration-efficient augmentation system for person-specific gaze estimation. It retains the explicit two-stream face/eye 3D Gaussian Splatting backbone introduced by [GazeGaussian](https://github.com/ucwxb/GazeGaussian), while replacing its convolutional rendering stage with a geometry-conditioned latent Diffusion Transformer (DiT). A learnable projection and frozen VAE connect rasterized Gaussian feature maps to latent tokens, while timestep and identity/expression conditioning are integrated through Adaptive Layer Normalization.

DiTGaze additionally introduces:

- an **Intermediate Gaze Sampler** that supplies yaw-pitch targets beyond the discrete training pairs;
- an **orthogonality constraint** that reduces interference between gaze, pose, and expression deformation embeddings; and
- an expanded **few-shot personalization evaluation** connecting gaze-redirection quality to downstream gaze-estimation performance.

On ETH-XGaze, DiTGaze reduces generator-level gaze error by 4.1% relative to GazeGaussian. In the repeated downstream comparison at nine real calibration images, it reduces subject-level estimator error by 0.73 degrees and improves 14 of 15 identities.

> [!IMPORTANT]
> This repository extends the official GazeGaussian implementation. To preserve compatibility with the upstream code and checkpoints, some filenames, class names, configuration names, and the Conda environment still contain `gazegaussian`. In this repository, the default configuration enables the DiTGaze renderer (`neural_renderer_type = "dit"`), frozen VAE interface, intermediate-gaze training, and orthogonality loss.

## News

- **[2026-10-29]** The revised and extended DiTGaze paper was released as arXiv v2.
- **[2026-10]** DiTGaze was accepted at WACV 2027.

## Environment Setup

We recommend Conda for dependency management.

```bash
git clone https://github.com/Abiram929/DiTGaze.git
cd DiTGaze

conda env create -f env.yaml
conda activate gazegaussian

# Python 3.8, CUDA 11.6 and PyTorch 1.12.0
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1120/download.html
pip install kaolin==0.13.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.0_cu116.html

# Install the differentiable rasterization submodules
cd submodules/diff-gaussian-rasterization
pip install -e .

cd ../simple-knn
pip install -e .

cd ../..
```

## Dataset and Upstream Initialization

Dataset preprocessing follows [GazeNeRF](https://github.com/AlessandroRuzzi/GazeNeRF) and [GazeGaussian](https://github.com/ucwxb/GazeGaussian). The processed datasets, configuration files, and initialization checkpoint below are hosted by the GazeGaussian authors on [Hugging Face](https://huggingface.co/ucwxb/GazeGaussian/). They are upstream assets rather than DiTGaze-specific releases; please follow their original licenses and terms of use.

Run the following commands from the repository root:

```bash
mkdir -p data checkpoint
cd data

# ETH-XGaze training set
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partaa
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partab
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partac
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partad
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partae
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partaf
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partag
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partah
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partai
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partaj
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partak
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partal
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partam
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.partan
cat ETH-XGaze.zip.part* > ETH-XGaze.zip
unzip ETH-XGaze.zip

# ETH-XGaze test set
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze_test.zip
unzip ETH-XGaze_test.zip

# Cross-dataset evaluation sets
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ColumbiaGaze.zip
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/MPIIFaceGaze.zip
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/gazecapture.zip
unzip ColumbiaGaze.zip
unzip MPIIFaceGaze.zip
unzip gazecapture.zip

# Required model configuration files
cd ../configs
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/config_models.zip
unzip config_models.zip

# Upstream GazeGaussian checkpoint used for initialization
cd ../checkpoint
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/gazegaussian_ckp.pth

cd ..
```

## Directory Structure

After preparing the datasets and initialization files, the relevant project structure is:

```text
DiTGaze/
├── data/
│   ├── ETH-XGaze/
│   ├── ETH-XGaze_test/
│   ├── ColumbiaGaze/
│   ├── MPIIFaceGaze/
│   └── gazecapture/
├── configs/
│   ├── config_files/
│   ├── config_models/
│   └── gazegaussian_options.py       # DiTGaze options retain the upstream name
├── checkpoint/
│   └── gazegaussian_ckp.pth          # Upstream initialization checkpoint
├── models/
│   ├── dit_block.py
│   ├── dit_neural_renderer.py
│   ├── vae_module.py
│   └── gaze_gaussian.py              # Two-stream backbone with DiTGaze renderer selection
├── scripts/train/
│   ├── train_meshhead.sh
│   └── train_gazegaussian.sh         # Compatibility filename; trains the configured DiTGaze model
├── train_meshhead.py
├── train_gazegaussian.py
└── README.md
```

## Training

Training consists of two stages:

1. pretrain the canonical mesh head, if a compatible mesh-head checkpoint is not already available;
2. train the DiTGaze model using the two-stream 3DGS backbone and the DiT renderer.

```bash
# Stage 1: canonical mesh head
bash scripts/train/train_meshhead.sh <GPU_ID>

# Stage 2: DiTGaze
# The inherited launcher name is retained for compatibility.
bash scripts/train/train_gazegaussian.sh <GPU_ID>
```

- `<GPU_ID>` is the CUDA device index.
- The default stage-two launcher initializes the compatible backbone from `checkpoint/gazegaussian_ckp.pth`.
- To initialize from your own mesh-head checkpoint, edit `scripts/train/train_gazegaussian.sh` and set `--load_meshhead_checkpoint` accordingly.
- The DiTGaze defaults are defined in `configs/gazegaussian_options.py`, including six DiT blocks, eight attention heads, a frozen four-channel VAE, intermediate-gaze training, and orthogonality regularization.

## Acknowledgments and Code Lineage

DiTGaze is implemented by extending [GazeGaussian](https://github.com/ucwxb/GazeGaussian), which provides the two-stream face/eye 3D Gaussian backbone, explicit Eye Rotation Field, data preparation pipeline, and compatible initialization checkpoint.

The upstream GazeGaussian implementation itself builds on ideas and code from [GazeNeRF](https://github.com/AlessandroRuzzi/GazeNeRF) and [Gaussian Head Avatar](https://github.com/YuelangX/Gaussian-Head-Avatar). We thank the authors of all three projects for releasing their work. Please retain their notices, follow their licenses, and cite the relevant papers when using this repository.

## Citation

If you find DiTGaze useful, please cite:

```bibtex
@inproceedings{panchalingam2027ditgaze,
  title     = {DiTGaze: Geometry-Conditioned Diffusion Augmentation for Few-Shot Gaze Estimation},
  author    = {Panchalingam, Abiram and Bodala, Indu P. and Middleton, Stuart E.},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

Please also cite the principal upstream works used by the implementation:

```bibtex
@inproceedings{wei2025gazegaussian,
  title     = {GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting},
  author    = {Wei, Xiaobao and Chen, Peng and Li, Guangyu and Lu, Ming and Chen, Hui and Tian, Feng},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {13293--13303},
  year      = {2025}
}

@inproceedings{ruzzi2023gazenerf,
  title     = {GazeNeRF: 3D-Aware Gaze Redirection With Neural Radiance Fields},
  author    = {Ruzzi, Alessandro and Shi, Xiangwei and Wang, Xi and Li, Gengyan and De Mello, Shalini and Chang, Hyung Jin and Zhang, Xucong and Hilliges, Otmar},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {9676--9685},
  year      = {2023}
}

@inproceedings{xu2024gaussianheadavatar,
  title     = {Gaussian Head Avatar: Ultra High-Fidelity Head Avatar via Dynamic Gaussians},
  author    = {Xu, Yuelang and Chen, Benwang and Li, Zhe and Zhang, Hongwen and Wang, Lizhen and Zheng, Zerong and Liu, Yebin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1931--1941},
  year      = {2024}
}
```
