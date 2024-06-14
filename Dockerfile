FROM registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.1
WORKDIR /workspace

RUN apt-get update && apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg 

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install timm==0.5.4 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

ADD build/Miniconda3-latest-Linux-x86_64.sh /root/miniconda3/miniconda_install.sh
RUN bash /root/miniconda3/miniconda_install.sh -b -u -p /root/miniconda3
RUN /root/miniconda3/bin/conda init bash
RUN /root/miniconda3/bin/conda install -y -c tartansandal conda-bash-completion
RUN rm -f /root/miniconda3/miniconda_install.sh
COPY build/condarc /root/.condarc
RUN /root/miniconda3/bin/conda create -y -n langchain python=3.11

# wget -c -t 100 -P /home/ https://github.com/AlibabaResearch/AdvancedLiterateMachinery/releases/download/v1.2.0-docX-release/DocXLayout_231012.pth
ADD models models
ADD cls cls
ADD det det
ADD latex latex
ADD ppocr ppocr
ADD rec rec
ADD struc struc
ADD table table
ADD tools tools
ADD docx_chain docx_chain
ADD config.yaml config.yaml
ADD custom_convert.py custom_convert.py
ADD utils.py utils.py
