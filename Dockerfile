FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    colmap \
    ffmpeg \  
    libsm6 \  
    libxext-dev \
    libxrender1 \
    netcat \ 
    unzip \ 
    vim \
    curl \  
    wget \
    ca-certificates \ 
    sudo \
    git \ 
    bzip2 \ 
    libx11-6 \
    python3-pyqt5 \
    python3-pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
 /bin/bash ~/miniconda.sh -b -p /home/user/miniconda \
 && chmod +x ~/miniconda.sh \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=true

# Create a Python 3.10 environment
RUN /home/user/miniconda/bin/conda create -y --name py39 python=3.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py39
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install -n base conda-build \
 && /home/user/miniconda/bin/conda clean -ya

RUN python3 -m pip --no-cache-dir install \
    tensorflow==2.13.* \
    jupyter \
    matplotlib \
    tensorboard \
    tqdm \
    plotly \
    opencv-python \
    git+https://github.com/tensorflow/examples.git

# CUDA 11.8-specific steps
RUN conda install -c conda-forge cudatoolkit=11.8 
RUN conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda install Pillow scipy pandas numpy pyyaml scikit-learn pytorch-lightning

# GDAL
RUN conda install -c conda-forge gdal
RUN conda install poppler
RUN conda clean -ya


# Set the default command to python3
CMD ["python3"]

# HOW TO RUN
# docker run --rm -it --init --gpus all -p 5000:8888 -p 6006-6015:6006-6015
# -v /tmp/.X11-unix:/tmp/.X11-unix -e QT_X11_NO_MITSHM=1 -e DISPLAY=$DISPLAY
# --volume="$PWD:/app" pytorch /bin/bash

# jupyter notebook --ip 0.0.0.0 (replace port 8888 with 5000)
