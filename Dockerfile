# Use nvidia/cuda image
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean

RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh
    
# set path to conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# setup conda virtual environment
COPY ./environment.yaml /tmp/environment.yaml
RUN conda update conda -y
RUN conda env create -f /tmp/environment.yaml

RUN echo "source activate base" >> ~/.bashrc
RUN echo "conda activate control" >> ~/.bashrc
ENV PATH=/opt/conda/envs/control/bin:$PATH
ENV CONDA_DEFAULT_ENV=control

WORKDIR /repo

CMD ["root/miniconda3/envs/control/bin/python", "rest_service.py", "config/tumor_application.ini"]