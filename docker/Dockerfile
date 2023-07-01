FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1

SHELL ["/bin/bash", "-c"]
WORKDIR /work

RUN apt-get update \
    && apt-get install -y curl sox unzip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /usr/local/src/*

RUN python -m pip install -U lightning torchaudio
