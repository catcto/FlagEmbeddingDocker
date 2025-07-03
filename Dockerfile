FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tar \
    wget \
    git \
    bash \
    vim

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install requirements
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict
RUN conda create -y -n flagembedding python=3.11
ENV CONDA_DEFAULT_ENV=flagembedding
ENV PATH="/root/miniconda3/bin:/opt/conda/envs/flagembedding/bin:${PATH}"
WORKDIR /root/CosyVoice
RUN pip install -U FlagEmbedding

# Set environment variables
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Run
COPY api.py .
CMD ["python", "api.py"]