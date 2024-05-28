import argparse
import os, sys
import time
import fastdeploy as fd
from multiprocessing import Pool
# from threading import Thread
import pickle
import shutil
import json
import numpy as np
import cv2

from latex.latex_rec import Latex2Text, sort_boxes
from table.utils import TableMatch
from utils import (
    convert_info_docx,
    convert_info_md,
    download_and_extract_models,
    read_image,
    read_yaml,
    save_structure_res,
    sorted_layout_boxes,
)
from table.table_det import Table
from docx_chain.modules.layout_analysis import LayoutAnalysis

device = "gpu"
device_id = 0
backend = 'default'

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

# "text", "title", "figure", "figure_caption", "table", "table_caption", "header", "footer", "reference", "equation"

def load_model(
    # layout_path = "/workspace/models/picodet_lcnet_x1_0_fgd_layout_cdla_infer",
    layout_path = "/workspace/models/docx_layout_231012.pth",
    num_class = 10,
    det_path = "/workspace/models/ch_PP-OCRv4_det_infer",
    rec_path = "/workspace/models/ch_PP-OCRv4_rec_infer",
    rec_bs = 16,
    formula_path = "/workspace/models/latex_rec.pth",
    anay_path = "/workspace/models/mfd.pt",
    ):
    # load layout model
    global layout_model

    # num_class = 10 if "cdla" in layout_path.lower() else 5
    # layout_model_file = os.path.join(layout_path, "model.pdmodel")
    # layout_params_file = os.path.join(layout_path, "model.pdiparams")
    # layout_option = fd.RuntimeOption()
    # layout_option.use_gpu(device_id)
    # layout_option.set_cpu_thread_num(10)
    # layout_model = fd.vision.ocr.StructureV2Layout(
    #     layout_model_file, layout_params_file, layout_option)
    # layout_model.postprocessor.num_class = num_class
    # global layout_labels
    # layout_labels = ["text", "title", "list", "table", "figure"]
    # if num_class == 10:
    #     layout_labels = [
    #         "text", "title", "figure", "figure_caption", "table", "table_caption",
    #         "header", "footer", "reference", "equation"
    #     ]
    
    layout_analysis_config = {
        'from_modelscope_flag': False,
        'model_path': layout_path,
    }
    layout_model = LayoutAnalysis(layout_analysis_config)

    # load det & rec model
    det_model_file = os.path.join(det_path, "inference.pdmodel")
    det_params_file = os.path.join(det_path, "inference.pdiparams")

    model_option = fd.RuntimeOption()

    if device.lower() == "gpu":
        model_option.use_gpu(device_id)
    
    if backend.lower() == "trt":
        model_option.use_trt_backend()

        model_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960])
        model_option.set_trt_cache_file(det_path + "/det_trt_cache.trt")
    
    elif backend.lower() == "pptrt":
        model_option.use_paddle_infer_backend()
        model_option.paddle_infer_option.collect_trt_shape = True
        model_option.paddle_infer_option.enable_trt = True

        model_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960])
        model_option.set_trt_cache_file(det_path)

    elif backend.lower() == "ort":
        model_option.use_ort_backend()

    elif backend.lower() == "paddle":
        model_option.use_paddle_infer_backend()

    global det_model
    det_model = fd.vision.ocr.DBDetector(det_model_file, det_params_file, runtime_option=model_option)

    rec_model_file = os.path.join(rec_path, "inference.pdmodel")
    rec_params_file = os.path.join(rec_path, "inference.pdiparams")
    rec_label_file = os.path.join(rec_path, "ppocr_keys_v1.txt")

    model_option = fd.RuntimeOption()

    if device.lower() == "gpu":
        model_option.use_gpu(device_id)
    
    if backend.lower() == "trt":
        model_option.use_tensorrt()

        model_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                        [rec_bs, 3, 48, 320],
                                        [rec_bs, 3, 48, 2304])
        model_option.set_trt_cache_file(rec_path + "/rec_trt_cache.trt")
    
    elif backend.lower() == "pptrt":
        model_option.use_paddle_infer_backend()
        model_option.paddle_infer_option.collect_trt_shape = True
        model_option.paddle_infer_option.enable_trt = True

        model_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                        [rec_bs, 3, 48, 320],
                                        [rec_bs, 3, 48, 2304])
        model_option.set_trt_cache_file(rec_path)

    elif backend.lower() == "ort":
        model_option.use_ort_backend()

    elif backend.lower() == "paddle":
        model_option.use_paddle_infer_backend()

    global rec_model
    rec_model = fd.vision.ocr.Recognizer(
        rec_model_file, rec_params_file, rec_label_file, runtime_option=model_option)
    
    det_model.preprocessor.max_side_len = 960
    det_model.postprocessor.det_db_thresh = 0.3
    det_model.postprocessor.det_db_box_thresh = 0.6
    det_model.postprocessor.det_db_unclip_ratio = 1.5
    det_model.postprocessor.det_db_score_mode = "slow"
    det_model.postprocessor.use_dilation = False
    
    global ppocr
    if "PP-OCRV4" in rec_path.upper():
        ppocr = fd.vision.ocr.PPOCRv4(
            det_model=det_model, cls_model=None, rec_model=rec_model)
        ppocr.rec_batch_size = rec_bs
    else:
        ppocr = fd.vision.ocr.PPOCRv3(
            det_model=det_model, cls_model=None, rec_model=rec_model)
        ppocr.rec_batch_size = rec_bs

    # load table model
    config = read_yaml(r'config.yaml')
    global table_structurer
    table_structurer = Table(config['Table'])

    global table_match
    table_match = TableMatch(filter_ocr_result=True)

    global latex_analyzer
    latex_analyzer = Latex2Text(
        formula_config = {'model_fp': formula_path},
        analyzer_config=dict(model_name='mfd',           model_type='yolov7', model_fp=anay_path), 
        device = device,
    )


def text_system(img, thre = 0.5):
    time1 = time.time()
    result = ppocr.predict(img)
    bbox = []
    rec = []
    for i in range(len(result.rec_scores)):
        if result.rec_scores[i] > thre:
            bbox.append(np.reshape(result.boxes[i], (4, 2)))
            rec.append((result.text[i],result.rec_scores[i]))
    return bbox, rec, time.time() - time1


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


def process_predict(pdf_info, save_folder, img_idx=0):
    print(f"pdf_info: {pdf_info} save_folder: {save_folder}")

    # if os.path.exists(save_folder):
    #     shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    file_basename = pdf_info[1]
    images = read_image(pdf_info[0])
    all_res = []
    start = time.time()
    for page_idx, image in enumerate(images):
        ori_im = image.copy()
        if layout_model is not None:
            layout_time1 = time.time()
            # result = layout_model.predict(ori_im)
            la_result = layout_model(ori_im)
            # print(la_result)
            layout_dets = la_result['layout_dets']
            layout_res = []
            for idx, layout_det in enumerate(layout_dets):
                category_id = layout_det['category_id']
                category_name = layout_model.mapping(category_id)
                category_name = label_conversion.get(category_name, category_name)
                poly = layout_det['poly']
                bbox = [poly[0], poly[1], poly[4], poly[5]]
                layout_res.append({'bbox': np.asarray(bbox), 'label': category_name})
            # print(layout_res)
            print('[layout] time cost: {:.2f}'.format(time.time() - layout_time1))

            # layout_res = []
            # for i in range(len(result.label_ids)):
            #     layout_res.append({'bbox': np.asarray(result.boxes[i]), 'label': layout_labels[result.label_ids[i]]})
            # print('layout res:', layout_res)
        else:
            h, w = ori_im.shape[:2]
            layout_res = [dict(bbox=None, label='table')]
        res_list = []
        for region_idx, region in enumerate(layout_res):
            res = ''
            if region['bbox'] is not None:
                x1, y1, x2, y2 = region['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi_img = ori_im[y1:y2, x1:x2, :]
            else:
                x1, y1, x2, y2 = 0, 0, w, h
                roi_img = ori_im

            roi_img_output = os.path.join("output", file_basename, "roi_img_v2")
            os.makedirs(roi_img_output, exist_ok=True)
            cv2.imwrite(os.path.join(roi_img_output, f"{page_idx}_{region_idx}_{region['label']}.jpg"), roi_img)

            if region['label'] == 'table':
                table_time1 = time.time()

                res = dict()
                structure_res, elapse = table_structurer(roi_img.copy())
                res['cell_bbox'] = structure_res[1].tolist()
                h, w = roi_img.shape[:2]
                dt_boxes, rec_res, ocr_time_dict = text_system(roi_img.copy())
                r_boxes = []
                for box in dt_boxes:
                    x_min = max(0, box[:, 0].min() - 1)
                    x_max = min(w, box[:, 0].max() + 1)
                    y_min = max(0, box[:, 1].min() - 1)
                    y_max = min(h, box[:, 1].max() + 1)
                    box = [x_min, y_min, x_max, y_max]
                    r_boxes.append(box)
                dt_boxes = np.array(r_boxes)
                if dt_boxes is None:
                    res = {'html': None}
                    region['label'] = 'figure'
                res['html'] = table_match(structure_res, dt_boxes, rec_res)
                print('[table] rec time: {:.2f}'.format(time.time() - table_time1))

            else:
                rec_time1 = time.time()
                wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)
                wht_im[y1:y2, x1:x2, :] = roi_img

                # wht_img_output = os.path.join("output", file_basename, "wht_img")
                # os.makedirs(wht_img_output, exist_ok=True)
                # cv2.imwrite(os.path.join(wht_img_output, f"{page_idx}_{region_idx}_{region['label']}.jpg"), wht_im)

                lax_img, mf_out = latex_analyzer.recognize_by_cnstd(wht_im, resized_shape=608)
                if mf_out == None:
                    filter_boxes, filter_rec_res, ocr_time_dict = text_system(wht_im)
                    style_token = [
                        '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                        '</b>', '<sub>', '</sup>', '<overline>',
                        '</overline>', '<underline>', '</underline>', '<i>',
                        '</i>'
                    ]
                    res = []
                    full_res_str = ""
                    for box, rec_res in zip(filter_boxes, filter_rec_res):
                        rec_str, rec_conf = rec_res
                        for token in style_token:
                            if token in rec_str:
                                rec_str = rec_str.replace(token, '')
                        box += [x1, y1]
                        res.append({
                            'text': rec_str,
                            'confidence': float(rec_conf),
                            'text_region': box.tolist()
                        })
                        full_res_str += rec_str
                    print('[{}] rec time: {:.2f}'.format(region['label'], time.time() - rec_time1))
                
                else:
                    lax_img = np.array(lax_img)
                    filter_boxes, filter_rec_res, ocr_time_dict = text_system(lax_img)
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
                    
                    print('[{}] rec time: {:.2f}'.format(region['label'], time.time() - rec_time1))

            res_list.append({
                'type': region['label'].lower(),
                'bbox': [x1, y1, x2, y2],
                'img': roi_img,
                'res': res,
                'img_idx': img_idx
            })
        end = time.time()
        # print(end - start)

        # save_structure_res(res_list, save_folder, file_basename)
        h, w, _ = image.shape
        res = sorted_layout_boxes(res_list, w)
        all_res += res
    convert_info_md(images, all_res, save_folder, f"{file_basename}")
    # save all_res to json
    # with open(os.path.join(os.path.join(save_folder, file_basename), f"res_v2.pkl"), "wb") as f:
    #     pickle.dump({'data': all_res, 'name': pdf_info[0]}, f)


if __name__ == '__main__':
    load_model()
    input_file = "input/科大讯飞2023半年报.pdf"
    process_predict((input_file, os.path.basename(input_file)), "output")
