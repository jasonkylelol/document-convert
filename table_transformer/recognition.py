import os, json
from PIL import Image, ImageDraw
import torch
from transformers import TableTransformerForObjectDetection
from torchvision import transforms

device="cuda"

model_root = "/root/huggingface/models"
model_path = os.path.join(model_root, "microsoft/table-transformer-structure-recognition-v1.1-all")

print(f"load from {model_path}")
structure_model = TableTransformerForObjectDetection.from_pretrained(model_path)
structure_model.to(device)


def pre_processing(image):
    class MaxResize(object):
        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
            return resized_image
    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pixel_values = structure_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    print(pixel_values.shape)
    return pixel_values


def post_processing(image, outputs):
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def outputs_to_objects(outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})
        return objects

    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, image.size, structure_id2label)
    print(cells)
    return cells


def visualize_cells(image, cells, file_basename):
    cropped_table_visualized = image.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="red")

    cropped_table_visualized.save(f"output/{file_basename}_cells.jpg")

    cell_json = json.dumps(cells, ensure_ascii=False)
    with open(f"output/{file_basename}_cells.json", "w+") as f:
        f.write(cell_json)

    os.makedirs(f"output/{file_basename}", exist_ok=True)
    cell_table_cropped = image.copy()
    for cell in cells:
        bbox = cell['bbox']
        cropped_img = cell_table_cropped.crop(bbox)
        label = cell['label'].replace(" ", "_")
        if label.strip() != "":
            target_file = f"cell_{label}-{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}.jpg"
            cropped_img.save(f'output/{file_basename}/{target_file}.jpg')


if __name__ == "__main__":
    file_path = '/root/github/document-convert/table_transformer/output/1_out_table0.jpg'
    file_basename = os.path.basename(file_path).rstrip(".jpg")
    image = Image.open(file_path).convert("RGB")

    inputs = pre_processing(image)
    with torch.no_grad():
        outputs = structure_model(inputs)
    cells = post_processing(image, outputs)

    visualize_cells(image, cells, file_basename)
    print("done")