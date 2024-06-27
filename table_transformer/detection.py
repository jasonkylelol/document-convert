import os
from transformers import AutoModelForObjectDetection
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

device="cuda"

model_root = "/root/huggingface/models"
model_path = os.path.join(model_root, "microsoft/table-transformer-detection")

print(f"load from {model_path}")
model = AutoModelForObjectDetection.from_pretrained(model_path, revision="no_timm")
model.to(device)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_detected_tables(img, det_tables, out_path=None):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                label='Table', hatch='//////', alpha=0.3),
                        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    if out_path is not None:
      plt.savefig(out_path, bbox_inches='tight', dpi=150)

    return fig


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

    detection_transform = transforms.Compose([
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    pixel_values = detection_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    return pixel_values


def post_processing(outputs, image):
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
    
    # update id2label to include "no object"
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"

    objects = outputs_to_objects(outputs, image.size, id2label)
    return objects


def iob(bbox1, bbox2):
    # Intersection Over Union (IoU) calculation
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(x1, x1_)
    yA = max(y1, y1_)
    xB = min(x2, x2_)
    yB = min(y2, y2_)

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    bbox1Area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2Area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(bbox1Area + bbox2Area - interArea)

    return iou


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def save_mark(image, objects, file_basename):
    fig = visualize_detected_tables(image, objects, out_path=f'output/{file_basename}_out.jpg')
    visualized_image = fig2img(fig)
    return visualized_image


def save_crop(image, objects, file_basename):
    tokens = []
    detection_class_thresholds = {
        "table": 0.5,
        "table rotated": 0.5,
        "no object": 10
    }
    crop_padding = 20

    tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=crop_padding)
    for idx, table_crop in enumerate(tables_crops):
        cropped_table = table_crop['image'].convert("RGB")
        cropped_table.save(f'output/{file_basename}_out_table{idx}.jpg')


if __name__ == "__main__":
    file_path = '/root/github/document-convert/output/科大讯飞2023半年报.pdf/fitz_img/1.jpg'
    file_basename = os.path.basename(file_path).rstrip(".jpg")
    image = Image.open(file_path).convert("RGB")

    inputs = pre_processing(image)
    with torch.no_grad():
        outputs = model(inputs)
    objects = post_processing(outputs, image)

    os.makedirs("output", exist_ok=True)
    
    save_mark(image, objects, file_basename)

    save_crop(image, objects, file_basename)

    print("done")

