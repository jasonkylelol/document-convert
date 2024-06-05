from paddle-npu:cann80RC1-ubuntu20-aarch64-gcc84-py310-dev

add github/FastDeploy/python/dist /build/dist
run pip3 install /build/dist/*

add github/FastDeploy/python/build/lib.linux-aarch64-cpython-310/fastdeploy/libs/third_libs/opencv/lib /usr/local/fastdeploy/libs/third_libs/opencv/lib

run rm -rf /build

WORKDIR /workspace

RUN apt-get update && apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx ffmpeg 

COPY github/document-convert/requirements.txt requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple