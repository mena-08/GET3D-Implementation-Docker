ARG BASE_IMAGE=pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
FROM $BASE_IMAGE

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# additional libraries
RUN pip install ninja imageio imageio-ffmpeg xatlas gdown

# install nvdiffrast
RUN pip install ../nvdiffrast/https://github.com/NVlabs/nvdiffrast/

# !!!INSTALL CLIP USING THE sh file

# install pytorch and required packages
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install meshzoo ipdb imageio gputil h5py point-cloud-utils imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0

# HDR image support
RUN imageio_download_bin freeimage
