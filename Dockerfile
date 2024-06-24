FROM ascendhub.huawei.com/public-ascendhub/ascend-pytorch:24.0.RC1-A2-1.11.0-ubuntu20.04

WORKDIR /workspace
USER root

RUN apt-get update && apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg
RUN /usr/bin/python -V

ADD build/Miniconda3-latest-Linux-aarch64.sh /root/miniconda3/miniconda_install.sh
RUN bash /root/miniconda3/miniconda_install.sh -b -u -p /root/miniconda3
RUN /root/miniconda3/bin/conda init bash
RUN /root/miniconda3/bin/conda install -y -c tartansandal conda-bash-completion
RUN rm -f /root/miniconda3/miniconda_install.sh
COPY build/condarc /root/.condarc
RUN /root/miniconda3/bin/conda create -y -n langchain python=3.11

COPY requirements.txt requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip cache purge
RUN rm -f requirements.txt

RUN sed -i 's/check_for_updates()/pass/g' /usr/local/python3.9.2/lib/python3.9/site-packages/albumentations/__init__.py
RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc
RUN echo -e 'export CLICOLOR="Yes"\nexport TERM=xterm\nexport LC_ALL=C.UTF-8\nexport LANG=C.UTF-8' >> /root/.bashrc

ADD build/vimrc /root/.vimrc
ADD pkgs/modelscope/utils/device.py /usr/local/python3.9.2/lib/python3.9/site-packages/modelscope/utils/device.py
ADD pkgs/modelscope/pipelines/cv/ocr_detection_pipeline.py /usr/local/python3.9.2/lib/python3.9/site-packages/modelscope/pipelines/cv/ocr_detection_pipeline.py
ADD pkgs/modelscope/pipelines/cv/ocr_recognition_pipeline.py /usr/local/python3.9.2/lib/python3.9/site-packages/modelscope/pipelines/cv/ocr_recognition_pipeline.py

ADD models/docx_layout_231012.pth models/docx_layout_231012.pth
ADD models/ocr-detection models/ocr-detection
ADD models/ocr-recognition models/ocr-recognition
ADD models/latex_rec.pth models/latex_rec.pth
ADD models/mfd.pt models/mfd.pt
ADD models/table_infer_en.onnx models/table_infer_en.onnx
ADD latex latex
ADD table table
ADD docx_chain docx_chain
ADD custom_convert.py custom_convert.py
ADD utils.py utils.py
