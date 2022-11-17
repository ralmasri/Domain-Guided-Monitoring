FROM ubuntu:20.04

ARG CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
ARG CONDA_SH=Miniconda3-py38_4.12.0-Linux-x86_64.sh
ARG CONDA_DIR=.tmp

RUN mkdir /Domain-Guided-Monitoring

COPY /data /Domain-Guided-Monitoring
COPY /notebooks /Domain-Guided-Monitoring
COPY /src /Domain-Guided-Monitoring
COPY /tests /Domain-Guided-Monitoring
COPY .gitignore /Domain-Guided-Monitoring
COPY environment.yml /Domain-Guided-Monitoring
COPY install.sh /Domain-Guided-Monitoring
COPY LICENSE /Domain-Guided-Monitoring
COPY main_refinement.py /Domain-Guided-Monitoring
COPY main.py /Domain-Guided-Monitoring
COPY Makefile /Domain-Guided-Monitoring
COPY README.md /Domain-Guided-Monitoring



RUN cd /Domain-Guided-Monitoring && \
    bash install.sh && \
    mkdir .tmp && \
    wget -nc $CONDA_URL && \
    bash ./$CONDA_DIR/$CONDA_SH -b -u -p ./$CONDA_DIR/miniconda3/ && \
    .tmp/miniconda3/bin/conda activate base && \
    .tmp/miniconda3/bin/conda env update --file environment.yml --prune && \
    .tmp/miniconda3/bin/conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia


