# ControlNetLib

## Download example model
```
cd models
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth?download=true
```

## Launching an interactive Docker Container

```
sudo docker build -t controlnet .
sudo docker run -v .:/repo -it --net=bridge -p 8000:8000 --gpus all controlnet
```

## Launching an inference container, e.g. for deployment

```
sudo docker build -t controlnet .
sudo docker run -v .:/repo --net=bridge -p 8000:8000  --gpus all controlnet /root/miniconda3/envs/control/bin/python /repo/rest_service.py /repo/models/control_sd15_canny.pth tumor
```