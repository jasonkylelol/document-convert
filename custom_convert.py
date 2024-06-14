import argparse
import os, sys
import time
import numpy as np
import cv2

# from latex.latex_rec import Latex2Text, sort_boxes
from table.utils import TableMatch
from utils import (
    convert_info_md,
    read_image,
    read_yaml,
    save_structure_res,
    sorted_layout_boxes,
)
from table.table_det import Table
from docx_chain.modules.layout_analysis import LayoutAnalysis
from docx_chain.modules.text_detection import TextDetection
from docx_chain.modules.text_recognition import TextRecognition

device = "gpu"

label_conversion = {
    "title": "title",
    "figure": "figure",
    "plain text": "text",
    "header": "header",
    "page number": "page_number",
    "footnote": "footer",
    "footer": "footer",
    "table": "table",
    "table caption": "table_caption",
    "figure caption": "figure_caption",
    "equation": "equation",
    "full column": "full_column",
    "sub column": "sub_column"
}

model_path = "/root/document-convert/models"

def load_model(
    layout_model_path = f"{model_path}/docx_layout_231012.pth",
    ocr_det_path = f"{model_path}/ocr-detection",
    ocr_rec_path = f"{model_path}/ocr-recognition",
    formula_path = f"{model_path}/latex_rec.pth",
    anay_path = f"{model_path}/mfd.pt",
    ):
    # layout model
    load_layout_model(layout_model_path)

    # ocr model
    load_ocr_model(ocr_det_path, ocr_rec_path)

    # load table model
    global table_structurer
    table_structurer = Table()

    global table_match
    table_match = TableMatch(filter_ocr_result=False)


def load_layout_model(layout_model_path):
    global layout_model
    layout_analysis_config = {
        'from_modelscope_flag': False,
        'model_path': layout_model_path,
    }
    layout_model = LayoutAnalysis(layout_analysis_config)


def load_ocr_model(ocr_det_path, ocr_rec_path):
    global text_detection_model, text_recognition_model

    text_detection_config = {
        'from_modelscope_flag': True,
        'model_path': ocr_det_path,
    }
    text_detection_model = TextDetection(text_detection_config)

    text_recognition_config = {
        'from_modelscope_flag': True,
        'model_path': ocr_rec_path,
    }
    text_recognition_model = TextRecognition(text_recognition_config)


def text_predict(img, thre = 0.5):
    # print(f"type of img: {type(img)}")
    det_result = text_detection_model(img)
    rec_result = text_recognition_model(img, det_result)

    bbox = []
    rec_res = []
    for i in range(det_result.shape[0]):
        poly = np.reshape(det_result[i], (4, 2))
        bbox.append(poly)

        rec = rec_result[i]
        rec_text = "".join(rec['text'])
        rec_res.append((rec_text, thre))
    
    # print(f"{bbox}\n{rec_res}\n")
    return bbox, rec_res


def layout_predict(ori_img):
    if layout_model is not None:
        layout_time1 = time.time()
        la_result = layout_model(ori_img)
        layout_dets = la_result['layout_dets']
        layout_res = []
        for idx, layout_det in enumerate(layout_dets):
            category_id = layout_det['category_id']
            category_name = layout_model.mapping(category_id)
            category_name = label_conversion.get(category_name, category_name)

            poly = layout_det['poly']
            bbox = [poly[0], poly[1], poly[4], poly[5]]
            layout_res.append({'bbox': np.asarray(bbox), 'label': category_name})
        print('[layout] time cost: {:.2f}'.format(time.time() - layout_time1))
    else:
        layout_res = [dict(bbox=None, label='table')]
    return layout_res


def extract_roi_img(ori_img, region):
    h, w = ori_img.shape[:2]
    if region['bbox'] is not None:
        w_offset, h_offset = 2, 0
        # if region['label'] == 'table':
        #     h_offset = 20
        x1, y1, x2, y2 = region['bbox']
        x1, y1, x2, y2 = int(x1) - w_offset, int(y1) - h_offset, \
            int(x2) + w_offset, int(y2)
        roi_img = ori_img[y1:y2, x1:x2, :]
    else:
        x1, y1, x2, y2 = 0, 0, w, h
        roi_img = ori_img
    return roi_img, h, w, x1, y1, x2, y2


def draw_boxes_and_save(output_path, img_name, roi_img, boxes):
    for box in boxes:
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[2]), int(box[3]))
        color = (0, 255, 0)  # Green color for the box
        thickness = 1  # Thickness of the box edges
        cv2.rectangle(roi_img, top_left, bottom_right, color, thickness)
    cv2.imwrite(os.path.join(output_path, img_name), roi_img)


def process_predict(pdf_info, save_folder, img_idx=0):
    print(f"pdf_info: {pdf_info} save_folder: {save_folder}")

    # if os.path.exists(save_folder):
    #     shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    file_basename = pdf_info[1]
    images = read_image(pdf_info[0])
    all_res = []
    for page_idx, image in enumerate(images):
        ori_img = image.copy()
        layout_res = layout_predict(ori_img)
        res_list = []
        for region_idx, region in enumerate(layout_res):
            roi_img, h, w, x1, y1, x2, y2 = extract_roi_img(ori_img, region)

            roi_img_output = os.path.join("output", file_basename, "roi_img")
            os.makedirs(roi_img_output, exist_ok=True)
            cv2.imwrite(os.path.join(roi_img_output, f"{page_idx}_{region_idx}_{region['label']}.jpg"), roi_img)

            if region['label'] == 'table':
                table_time1 = time.time()

                res = dict()
                structure_res, elapse = table_structurer(roi_img.copy())
                res['cell_bbox'] = structure_res[1].tolist()
                dt_boxes, rec_res = text_predict(roi_img.copy())
                r_boxes = []
                offset = 1
                for box in dt_boxes:
                    x_min = max(0, box[:, 0].min() - offset)
                    x_max = min(w, box[:, 0].max() + offset)
                    y_min = max(0, box[:, 1].min() - offset)
                    y_max = min(h, box[:, 1].max() + offset)
                    box = [x_min, y_min, x_max, y_max]
                    r_boxes.append(box)

                table_img_output = os.path.join("output", file_basename, "table_img")
                os.makedirs(table_img_output, exist_ok=True)
                draw_boxes_and_save(table_img_output, f"{page_idx}_{region_idx}_table_structure.jpg",
                    roi_img.copy(), res['cell_bbox'])

                # print(f"structure_res: {structure_res}")

                dt_boxes = np.array(r_boxes)
                if dt_boxes is None:
                    res = {'html': None}
                    region['label'] = 'figure'
                res['html'] = table_match(structure_res, dt_boxes, rec_res)
                print('[table] rec time: {:.2f}'.format(time.time() - table_time1))

            else:
                rec_time1 = time.time()
                wht_im = np.ones(ori_img.shape, dtype=ori_img.dtype)
                wht_im[y1:y2, x1:x2, :] = roi_img

                # wht_img_output = os.path.join("output", file_basename, "wht_img")
                # os.makedirs(wht_img_output, exist_ok=True)
                # cv2.imwrite(os.path.join(wht_img_output, f"{page_idx}_{region_idx}_{region['label']}.jpg"), wht_im)

                filter_boxes, filter_rec_res = text_predict(wht_im)
                style_token = [
                    '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                    '</b>', '<sub>', '</sup>', '<overline>',
                    '</overline>', '<underline>', '</underline>', '<i>',
                    '</i>'
                ]
                res = []
                full_res_str = ""
                for box, rec_res in zip(filter_boxes, filter_rec_res):
                    rec_str, rec_score = rec_res
                    for token in style_token:
                        if token in rec_str:
                            rec_str = rec_str.replace(token, '')
                    box += [x1, y1]
                    res.append({
                        'text': rec_str,
                        'confidence': float(rec_score),
                        'text_region': box.tolist()
                    })
                    full_res_str += rec_str
                print('[{}] rec time: {:.2f}'.format(region['label'], time.time() - rec_time1))

            res_list.append({
                'type': region['label'].lower(),
                'bbox': [x1, y1, x2, y2],
                'img': roi_img,
                'res': res,
                'img_idx': img_idx
            })
        
        h, w, _ = image.shape
        res = sorted_layout_boxes(res_list, w)
        all_res += res

    save_structure_res(all_res, save_folder, file_basename)
    convert_info_md(images, all_res, save_folder, f"{file_basename}")
    # save all_res to json
    # with open(os.path.join(os.path.join(save_folder, file_basename), f"res_v2.pkl"), "wb") as f:
    #     pickle.dump({'data': all_res, 'name': pdf_info[0]}, f)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("[custom_convert.py input_file] Exactly one input file is required")
    
    input_file = sys.argv[1]
    
    if not input_file:
        raise ValueError("[custom_convert.py input_file] input file cannot be empty")

    load_model()
    process_predict((input_file, os.path.basename(input_file)), "output")
