# Hardware
Deploy Azure VM - Ubuntu 20.04 - NC6v3

# Installation
```bash
# Install CUDA drivers
sudo apt-get update
sudo apt-get install -y build-essential linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda

# Setup Docker with nvidia Drivers
curl https://get.docker.com | sh
sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Configure Docker
sudo mkdir -p /mnt/lib/docker  # use of larger mounted storage space is required
sudo systemctl stop docker 
sudo systemctl edit docker --full
# change ExecStart line -> ExecStart=/usr/bin/dockerd -g /mnt/lib/docker -H fd:// --containerd=/run/containerd/containerd.sock
sudo systemctl daemon-reload
sudo usermod -aG docker $USER
newgrp docker
```

# Getting Source and Model Weights
```bash
sudo mkdir /mnt/images  # image storage path

# from game-paint/server directory
pushd server/app/model
git clone https://github.com/openai/CLIP
git clone https://github.com/CompVis/taming-transformers

mkdir checkpoints
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
popd
```

# Build Docker Image
```bash
# build docker image
cd server
docker build -t game-paint .

# verify setup
docker run --gpus all --rm game-paint python -c "import torch; assert torch.cuda.is_available(); print(f'num_gpus: {torch.cuda.device_count()}'); print(torch.cuda.get_device_name(0))"
# should return
# num_gpus: 1
# Tesla V100-PCIE-16GB
```

# Test
```bash
cd game-paint/server
sudo docker run --gpus all -v `pwd`/app:/app -v /mnt/images:/mnt/images -it game-paint python
>>> from vqgan_clip import generate, load_perceptor, load_model
>>> model = load_model()
>>> perceptor = load_perceptor()
>>> generate(model, perceptor, prompts="a sunny bedroom | unreal engine", output_path="output.png")
```

# Run Server
```bash
cd game-paint/server
sudo docker run --gpus all -v `pwd`/app:/app -v /mnt/images:/images -p 8000:8000 -d -t game-paint
```
