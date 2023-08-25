FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0/index.html

# Install MMDetection
RUN pip install mmdet==2.14.0

# Install MMSeg
RUN pip install mmsegmentation==0.14.1

# Install MMDetection3D requirements
RUN conda clean --all
RUN git clone https://github.com/tusen-ai/SST.git /sst
WORKDIR /sst
RUN git checkout 8b795c40632c3868b091dc032fff7099c0d21bc5
RUN pip install llvmlite --ignore-installed
RUN pip install pyyaml --ignore-installed
RUN pip install -r requirements/build.txt

RUN pip install --no-cache-dir -e .

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

# Install CLIP dependecies and CLIP
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

# Install our own requirements
RUN pip install \
    wandb \
    pipenv \
    pytorch_lightning==1.6.2 

RUN pip install "numpy<1.24.0"

RUN pip install opencv-python==4.5.5.64