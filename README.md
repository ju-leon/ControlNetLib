# ControlNetLib

## Download example model
```
cd models
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth?download=true
```

## Launching a Docker Container

```
sudo docker build -t controlnet .
sudo docker run -v .:/repo -it --net=bridge -p 8000:8000 --gpus all controlnet
```