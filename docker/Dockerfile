FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel 
ENV LC_ALL=C.UTF-8
RUN pip install transformers pandas black pylint sklearn tqdm fairseq
EXPOSE 8888
WORKDIR /workspace 
