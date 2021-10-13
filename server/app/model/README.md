# Hardware
Deploy Azure VM - Ubuntu 20.04 - NC6v3

# Installation
```bash
# Install gcc ?
sudo apt-get update
sudo apt-get install -y build-essential

# Install CUDA drivers ?
sudo apt-get install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda

# Setup Docker
curl https://get.docker.com | sh
sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl stop docker
sudo systemctl edit docker --full
sudo mkdir -p /mnt/lib/docker  # use of larger mounted storage space is required
# change ExecStart line -> ExecStart=/usr/bin/dockerd -g /mnt/lib/docker -H fd:// --containerd=/run/containerd/containerd.sock
sudo systemctl restart docker
```

# Getting Source and Model Weights
```bash
cd game-paint/app/model
git clone 'https://github.com/openai/CLIP'
git clone 'https://github.com/CompVis/taming-transformers'

sudo mkdir /mnt/images  # image storage path
mkdir checkpoints
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```

# Build Docker Image
```bash
# build docker image
sudo docker build -t game-paint .

# verify setup
sudo docker run --gpus all --rm game-paint python -c "import torch; assert torch.cuda.is_available(); print(f'num_gpus: {torch.cuda.device_count()}'); print(torch.cuda.get_device_name(0))"
```

# Test
```bash
python
>>> from vqgan_clip import generate
>>> model = load_model()
>>> perceptor = load_perceptor()
>>> generate(model, perceptor, prompts="a sunny bedroom | unreal engine", output_path="output.png")
```

