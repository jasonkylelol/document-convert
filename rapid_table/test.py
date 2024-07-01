from pathlib import Path

from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable
from rapid_table import RapidTable, VisTable
from PIL import Image, ImageDraw

# RapidTable类提供model_path参数，可以自行指定上述2个模型，默认是en_ppstructure_mobile_v2_SLANet.onnx
# table_engine = RapidTable(model_path='ch_ppstructure_mobile_v2_SLANet.onnx')
table_model_path = "/root/huggingface/models/paddle-tsr/ch_ppstructure_mobile_v2.0_SLANet_infer/ch_ppstructure_mobile_v2_SLANet.onnx"
table_engine = RapidTable(model_path=table_model_path)
ocr_engine = RapidOCR()
viser = VisTable()

img_path = '/root/github/document-convert/table_transformer/output/1_out_table0.jpg'

ocr_result, _ = ocr_engine(img_path)
table_html_str, table_cell_bboxes, elapse = table_engine(img_path, ocr_result)

save_dir = Path("./output/")
save_dir.mkdir(parents=True, exist_ok=True)

save_html_path = save_dir / f"{Path(img_path).stem}.html"
save_drawed_path = save_dir / f"vis_{Path(img_path).name}"

viser(img_path, table_html_str, save_html_path, table_cell_bboxes, save_drawed_path)

# print(ocr_result)
img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img)
for obj in ocr_result:
    bbox = obj[0]
    top_left = tuple(bbox[0])
    bottom_right = tuple(bbox[2])
    draw.rectangle([top_left, bottom_right], outline="green")
img.save(f"output/ocr_cells_{Path(img_path).name}.jpg")

# print(table_cell_bboxes)
img2 = img.copy()
draw2 = ImageDraw.Draw(img2)
for bbox in table_cell_bboxes:
    top_left = (bbox[0], bbox[1])
    bottom_right = (bbox[4], bbox[5])
    print([top_left, bottom_right])
    draw2.rectangle([top_left, bottom_right], outline="green")
img2.save(f"output/table_cells_{Path(img_path).name}.jpg")

print(table_html_str, "\n", f"elapse: {elapse}")