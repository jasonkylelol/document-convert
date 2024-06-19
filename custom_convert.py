import argparse
import os, sys
import time
import numpy as np
import cv2

from utils import (
    convert_info_md,
    read_image,
    read_yaml,
    save_structure_res,
    sorted_layout_boxes,
)
from table.utils import TableMatch
from table.table_det import Table
from docx_chain.modules.layout_analysis import LayoutAnalysis
from docx_chain.modules.text_detection import TextDetection
from docx_chain.modules.text_recognition import TextRecognition
from latex.latex_rec import Latex2Text, sort_boxes

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

model_path = "/workspace/models"

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
    load_table_model()
    # load latex model
    # load_latex_model(formula_path, anay_path)


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


def load_table_model():
    global table_structurer
    table_structurer = Table()

    global table_match
    table_match = TableMatch(filter_ocr_result=False)


def load_latex_model(formula_path, anay_path):
    global latex_analyzer
    latex_analyzer = Latex2Text(
        formula_config = {'model_fp': formula_path},
        analyzer_config=dict(model_name='mfd', model_type='yolov7', model_fp=anay_path), 
        device = device,
    )


def text_extraction(img, thre = 0.5):
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
        w_offset, h_offset = 6, 4
        # if region['label'] == 'table':
        #     h_offset = 20
        x1, y1, x2, y2 = region['bbox']
        x1, y1, x2, y2 = int(x1) - w_offset, int(y1) - h_offset, \
            int(x2) + w_offset, int(y2) + h_offset
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


def latex_predict(ori_img, x1, y1, x2, y2, roi_img,
    file_basename=None, page_idx=None, region_idx=None, label=None):
    wht_im = np.ones(ori_img.shape, dtype=ori_img.dtype)
    wht_im[y1:y2, x1:x2, :] = roi_img

    if file_basename:
        wht_img_output = os.path.join("output", file_basename, "wht_img")
        os.makedirs(wht_img_output, exist_ok=True)
        cv2.imwrite(os.path.join(wht_img_output, f"{page_idx}_{region_idx}_{label}.jpg"), wht_im)

    latex_img, mf_out = latex_analyzer.recognize_by_cnstd(wht_im, resized_shape=608)
    if not mf_out:
        return None
    latex_img = np.array(latex_img)
    filter_boxes, filter_rec_res = text_extraction(latex_img)
    style_token = [
        '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
        '</b>', '<sub>', '</sup>', '<overline>',
        '</overline>', '<underline>', '</underline>', '<i>',
        '</i>'
    ]
    st_res =  []
    for lat in mf_out:
        lat['position']+=[x1,y1]
        st_res.append({
            'text': lat['text'],
            'confidence': float(0.8),
            'position': lat['position']#.tolist()
            })
    for box, rec_res in zip(filter_boxes, filter_rec_res):
        rec_str, rec_conf = rec_res
        for token in style_token:
            if token in rec_str:
                rec_str = rec_str.replace(token, '')
        box += [x1, y1]
        st_res.append({
            'text': rec_str,
            'confidence': float(rec_conf),
            'position': box#.tolist()
        })

    st_res = sort_boxes(st_res, key='position')

    res = []
    for i in st_res:
        i = i[0]
        i['text_region'] = i['position'].tolist()
        del i['position']
        res.append(i)
    return res


def table_predict(roi_img, region, h, w, file_basename=None, page_idx=None, region_idx=None):
    res = dict()
    structure_res, elapse = table_structurer(roi_img.copy())
    res['cell_bbox'] = structure_res[1].tolist()
    dt_boxes, rec_res = text_extraction(roi_img.copy())
    r_boxes = []
    offset = 1
    for box in dt_boxes:
        x_min = max(0, box[:, 0].min() - offset)
        x_max = min(w, box[:, 0].max() + offset)
        y_min = max(0, box[:, 1].min() - offset)
        y_max = min(h, box[:, 1].max() + offset)
        box = [x_min, y_min, x_max, y_max]
        r_boxes.append(box)

    if file_basename:
        table_img_output = os.path.join("output", file_basename, "table_img")
        os.makedirs(table_img_output, exist_ok=True)
        draw_boxes_and_save(table_img_output, f"{page_idx}_{region_idx}_table_structure.jpg",
            roi_img.copy(), res['cell_bbox'])

    dt_boxes = np.array(r_boxes)
    if dt_boxes is None:
        res = {'html': None}
        region['label'] = 'figure'
    res['html'] = table_match(structure_res, dt_boxes, rec_res)
    return res


def text_predict(ori_img, x1, y1, x2, y2, roi_img,
    file_basename=None, page_idx=None, region_idx=None, label=None):
    wht_im = np.ones(ori_img.shape, dtype=ori_img.dtype)
    wht_im[y1:y2, x1:x2, :] = roi_img

    if file_basename:
        wht_img_output = os.path.join("output", file_basename, "wht_img")
        os.makedirs(wht_img_output, exist_ok=True)
        cv2.imwrite(os.path.join(wht_img_output, f"{page_idx}_{region_idx}_{label}.jpg"), wht_im)

    filter_boxes, filter_rec_res = text_extraction(wht_im)
    res = []
    for box, rec_res in zip(filter_boxes, filter_rec_res):
        rec_str, rec_score = rec_res
        box += [x1, y1]
        res.append({
            'text': rec_str,
            'confidence': float(rec_score),
            'text_region': box.tolist()
        })
    return res


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
            label = region['label']

            # roi_img_output = os.path.join("output", file_basename, "roi_img")
            # os.makedirs(roi_img_output, exist_ok=True)
            # cv2.imwrite(os.path.join(roi_img_output, f"{page_idx}_{region_idx}_{label}.jpg"), roi_img)

            if label == 'table':
                table_stime = time.time()
                res = table_predict(roi_img, region, h, w,
                    # file_basename, page_idx, region_idx,
                    )
                print('[table] rec time: {:.2f}'.format(time.time() - table_stime))
            
            elif label == 'equation':
                res = None
                # latex_stime = time.time()
                # res = latex_predict(ori_img, x1, y1, x2, y2, roi_img,
                #     file_basename, page_idx, region_idx, label)
                # if not res:
                #     continue
                # print('[latex] rec time: {:.2f}'.format(time.time() - latex_stime))

            else:
                rec_stime = time.time()
                res = text_predict(ori_img, x1, y1, x2, y2, roi_img,
                    # file_basename, page_idx, region_idx, label,
                    )
                print('[{}] rec time: {:.2f}'.format(label, time.time() - rec_stime))

            res_list.append({
                'type': label.lower(),
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
