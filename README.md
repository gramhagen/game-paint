# Overview

This project uses AI to generate art based on text prompts which can then be placed in your room using an AR application on android.

There are two main components to setup
1. The model server running the AI art generation service
1. The android app to get user input and render the image in AR

# Architecture

## Unity
The Unity application has two scenes
1. UI - responsible for collecting the text input. The main class which manages communication with the back end servers is the ModelService.cs file
2. ARArtScene - responsible for rendering the art. The main class which controls the image placement is TapToPlaceImage.cs

## Model
The model service uses two models Vector Quantized Generative Adverserial Network (VQGAN) for image generation and Contrastive Language Image Pre-training (CLIP) to guide VQGAN into matching the text description. These are wrapped into a REST based service using  the FastAPI framwork.

The server and model setup is handled using a Docker image based on NVIDIA's PyTorch base image and leveraging code from NerdyRodent's github [VQGAN+CLIP](https://github.com/nerdyrodent/VQGAN-CLIP) repo.

# Installation

## Model Server
1. See model server [README.md](server/README.md)
1. Adjust TOKEN constant in server/app/app.py to be a unique string


## Android App
1. Install [Unity Hub](https://public-cdn.cloud.unity3d.com/hub/prod/UnityHubSetup.exe?_ga=2.156323643.162358957.1634581825-600115226.1633117491) (not beta version)
1. Then download and install Unity Version 2020.3.1 - [Download Archive](https://unity3d.com/get-unity/download/archive)
1. Open project (adjust SERVER_TOKEN and SERVER_URL constants to match deployed back-end server configuration)
1. Connect your phone to your computer with USB debugging enabled and choose File->Build and Run to build and run the game on your device