### This repository was modified by Omar Mena, according to the license, to dockerize and generalize the use of the GET3D paper with a CLIP implementation.

# Abstract of the project: 
This project presents a novel approach to integrating CLIP(Contrastive Language-Image Pre-training) into the inference stage of a generative 3D modeling system based on GET3D. By leveraging CLIP’s ability to align textual descriptions with visual representations, we facilitate the generation of 3D models that exhibit high visual quality, as we are using an updated and upgraded ShapeNet dataset that contains more accurate information and aligns with user-provided text inputs. Our approach provides controllable and customization of the generated 3D content, allowing users to access 3D ad-hoc content. Through ablation studies, we demonstrate the effectiveness of our CLIP integration in improving the relevance and quality of the generated 3D models compared to the original GET3D model. This work contributes to the advancement of language-guided 3D content generation and has potential applications in various domains such as digital art, gaming, and design.

## Results of th project
[image info](./pictures/image.png)

# Docker Setup Guide for Linux and Windows (WSL)

## Docker Setup on Linux
Docker is a powerful platform for developing, shipping, and running applications. Below, you will find detailed steps on how to set up Docker on both Linux and Windows using the Windows Subsystem for Linux (WSL).

### Uninstall Old Versions
Older versions of Docker were called `docker`, `docker.io`, or `docker-engine`. If these are installed, uninstall them along with associated dependencies.

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

### Update the apt package index and install packages to allow apt to use a repository over HTTPS:
 ```bash
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```

### Add Docker’s official GPG key:
```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

### Set up the stable repository:
```bash
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

### Update the apt package index, and install the latest version of Docker Engine and containerd:
```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

### Verify that Docker Engine is installed correctly by running the hello-world image:
```bash
sudo docker run hello-world
```

/////////////////////
## Docker Setup on Windows with WLS

### Open PowerShell as Administrator and run:
```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

### Enable Virtual Machine feature:
 ```bash
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

### Set WSL 2 as your default version:
```bash
wsl --set-default-version 2
```

### Downloading Docker
1. Download Docker Desktop for Windows from the Docker Hub.

2. Run the installer and ensure the "Enable WSL 2 Features" option is checked.

3. After installation, Docker Desktop prompts to log out and log back in; ensure to complete this step.

4. Docker Desktop automatically integrates with WSL. Verify by opening your Linux distribution via WSL and typing:
```bash
docker run hello-world
```


# How to setup blender docker with GPUs to render ShapeNet V2
## Prerrequisites
- NVIDIA GPU Drivers and Docker: You must have NVIDIA drivers installed to enable GPU usage within Docker containers. Refer to the NVIDIA Docker GitHub for installation instructions. -> https://github.com/NVIDIA/nvidia-container-toolkit
- If you installed everything correctly, you will run the following command and be able to use Blender on Docker with GPU's
```bash
docker run --gpus all -ti -v "%cd%":/home/project/blender -e CYCLES_DEVICE=CUDA nytimes/blender:2.90-gpu-ubuntu18.04
```

## Just a few more steps...
Now we just need install the last libraries, for that I included them in form of shell files so you can just run them
```bash
sh install_get3d.sh
sh install_clip.sh
sh install_kaolin.sh
sh install_nvidiadiffrast.sh
```

# How to setup nvidia docker with GPUs
This part is directly related to the usage of the project.
Once you have followed the prerrequisites above, we can start building the project's docker 
```bash
#From the root folder
cd docker
sh make_image.sh get3d:vn .

#after building the image we can run it using the GPU's
docker run --gpus all -ti -v "%cd%":/home/project get3d:vn

#we can check if all the modules and cuda are available by using without erros
nvcc --version
```



## GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images

Abstract of the project: *As several industries are moving towards modeling massive 3D virtual worlds,
the need for content creation tools that can scale in terms of the quantity, quality, and
diversity of 3D content is becoming evident. In our work, we aim to train performant 3D
generative models that synthesize textured meshes which can be directly consumed by 3D
rendering engines, thus immediately usable in downstream applications. Prior works on 3D
generative modeling either lack geometric details, are limited in the mesh topology they
can produce, typically do not support textures, or utilize neural renderers in the
synthesis process, which makes their use in common 3D software non-trivial. In this work,
we introduce GET3D, a Generative model that directly generates Explicit Textured 3D meshes
with complex topology, rich geometric details, and high fidelity textures. We bridge
recent success in the differentiable surface modeling, differentiable rendering as well as
2D Generative Adversarial Networks to train our model from 2D image collections. GET3D is
able to generate high-quality 3D textured meshes, ranging from cars, chairs, animals,
motorbikes and human characters to buildings, achieving significant improvements over
previous methods.*

## Requirements

* We recommend Linux for performance and compatibility reasons.
* 1 &ndash; 8 high-end NVIDIA GPUs. We have done all testing and development using V100 or A100
  GPUs.
* 64-bit Python 3.8 and PyTorch 1.9.0. See https://pytorch.org for PyTorch install
  instructions.
* CUDA toolkit 11.1 or later.  (Why is a separate CUDA toolkit installation required? We
  use the custom CUDA extensions from the StyleGAN3 repo. Please
  see [Troubleshooting](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary))
  .
* We also recommend to install Nvdiffrast following instructions
  from [official repo](https://github.com/NVlabs/nvdiffrast), and
  install [Kaolin](https://github.com/NVIDIAGameWorks/kaolin).
* We provide a [script](./install_get3d.sh) to install packages.

### Server usage through Docker

- Build Docker image

```bash
cd docker
chmod +x make_image.sh
./make_image.sh get3d:v1
```

- Start an interactive docker
  container: `docker run --gpus device=all -it --rm -v YOUR_LOCAL_FOLDER:MOUNT_FOLDER -it get3d:v1 bash`

## Preparing datasets

GET3D is trained on synthetic dataset. We provide rendering scripts for Shapenet. Please
refer to [readme](./render_shapenet_data/README.md) to download shapenet dataset and
render it.

## Train the model

#### Clone the gitlab code and necessary files:

```bash
cd YOUR_CODE_PATH
git clone git@github.com:nv-tlabs/GET3D.git
cd GET3D; mkdir cache; cd cache
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
```

#### Train the model

```bash
cd YOUR_CODE_PATH 
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

- Train on the unified generator on cars, motorbikes or chair (Improved generator in
  Appendix):

```bash
python train_3d.py --outdir=PATH_TO_LOG --data=PATH_TO_RENDER_IMG --camera_path PATH_TO_RENDER_CAMERA --gpus=8 --batch=32 --gamma=40 --data_camera_mode shapenet_car  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0
python train_3d.py --outdir=PATH_TO_LOG --data=PATH_TO_RENDER_IMG --camera_path PATH_TO_RENDER_CAMERA --gpus=8 --batch=32 --gamma=80 --data_camera_mode shapenet_motorbike  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0
python train_3d.py --outdir=PATH_TO_LOG --data=PATH_TO_RENDER_IMG --camera_path PATH_TO_RENDER_CAMERA --gpus=8 --batch=32 --gamma=400 --data_camera_mode shapenet_chair  --dmtet_scale 0.8  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0
```

If want to debug the model first, reduce the number of gpus to 1 and batch size to 4 via:

```bash
--gpus=1 --batch=4
```

## Inference

### Inference on a pretrained model for visualization

- Download pretrained model from [here](https://drive.google.com/drive/folders/1oJ-FmyVYjIwBZKDAQ4N1EEcE9dJjumdW?usp=sharing).
- Inference could operate on a single GPU with 16 GB memory.

```bash
python train_3d.py --outdir=save_inference_results/shapenet_car  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_car  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain MODEL_PATH
python train_3d.py --outdir=save_inference_results/shapenet_chair  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_chair  --dmtet_scale 0.8  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain MODEL_PATH
python train_3d.py --outdir=save_inference_results/shapenet_motorbike  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_motorbike  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain MODEL_PATH
```

- To generate mesh with textures, add one option to the inference
  command: `--inference_to_generate_textured_mesh 1`

- To generate the results with latent code interpolation, add one option to the inference
  command: `--inference_save_interpolation 1`

#Results
## PLACEHOLDER OF IMAGES NOW

### Evaluation metrics

##### Compute COV & MMD scores for LFD & CD

- First generate 3D objects for evaluation, add one option to the inference
  command: `--inference_generate_geo 1`
- Following [README](./evaluation_scripts/README.md) to compute metrics.
