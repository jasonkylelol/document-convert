image=pdf-document-convert:v0.1
gpus="device=0"

docker run -it --rm --cpus 16 -m 32G --gpus $gpus -h doc-converter --name doc-converter \
	-v /home/lm/github/document-convert:/workspace \
	$image bash