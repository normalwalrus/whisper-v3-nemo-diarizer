# docker build -t asr-eval/whisper-hf:1.0.0 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install libsndfile1 (linux soundfile package)
RUN apt-get clean \
    && apt-get update \ 
    && apt-get install -y gcc g++ libsndfile1 ffmpeg sox wget git 
    #&& rm -rf /var/lib/apt/lists/*

RUN apt-get update -y \
    && apt-get install -y python3 python3-pip \
    && python -m pip install --no-cache-dir poetry==1.8.3

ARG NEMO_VERSION=1.21.0

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir Cython==0.29.35 && \
    pip3 install --no-cache-dir nemo_toolkit[asr]==${NEMO_VERSION}

WORKDIR /opt/app-root

ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# COPY poetry.lock pyproject.toml ./
# RUN poetry install

RUN ["python", "-c", "from nemo.collections.asr.models.msdd_models import NeuralDiarizer; NeuralDiarizer.from_pretrained('diar_msdd_telephonic')"]
RUN ["python", "-c", "from nemo.collections.asr.models import EncDecSpeakerLabelModel; EncDecSpeakerLabelModel.from_pretrained('nvidia/speakerverification_en_titanet_large')"]

#ENTRYPOINT ["bash"]
