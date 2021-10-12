# Dev guide to run the solution

## Server

- Build server docker image

        sudo docker build -t image-generation -f server/Dockerfile .


- Run server locally

    Without gpu

        sudo docker run --rm -it -p 8886:8886 image-generation:latest

    With gpu

        sudo docker run --gpus all --rm -it -p 8886:8886 image-generation:latest
