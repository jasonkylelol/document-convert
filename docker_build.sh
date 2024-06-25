TARGET=document-convert:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.1

docker build -t $TARGET .
