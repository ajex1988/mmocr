from mmocr.utils.ocr import MMOCR
from datetime import datetime
from tqdm import tqdm
import os


def run_det_exp(model_name_list, dir_list, exp_dir):
    '''
    run a text detection experiment
    '''
    for model_name in tqdm(model_name_list):
        model = MMOCR(det=model_name, recog=None)
        for d in dir_list:
            d_base = os.path.basename(d)
            output_dir = os.path.join(exp_dir, d_base)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            run_det_batch(model=model, model_name=model_name, img_dir=d, output_dir=output_dir)


def run_det_batch(model, model_name, img_dir, output_dir):
    '''
    run text detection for all the images in the img_dir
    '''
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    img_list = [os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir) if os.path.splitext(img_file)[1] in extensions]
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        output_img_path = os.path.join(output_dir, os.path.splitext(img_name)[0]+"_"+model_name+"_"+".png")
        run_det_save(model=model, img_path=img_path, output_img_path=output_img_path)


def run_det_save(model, img_path, output_img_path):
    '''
    run text detection on an image and save the detection result
    '''
    result = model.readtext(img=img_path, output=output_img_path)


def main_det():
    model_name_list = ["DB_r18",
                  "DB_r50",
                  "DBPP_r50",
                  "DRRG",
                  "FCE_IC15",
                  "FCE_CTW_DCNv2",
                  "MaskRCNN_CTW",
                  "MaskRCNN_IC15",
                  "MaskRCNN_IC17",
                  "PANet_CTW",
                  "PANet_IC15",
                  "PS_CTW",
                  "PS_IC15",
                  "Tesseract",
                  "TextSnake"]
    dir_list = ["D:/Projects/mmocr/data/dms/keyframes",
                "D:/Projects/mmocr/data/dms/LQ_keyframes",
                "D:/Projects/mmocr/data/dms/NQ_keyframes"]
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    exp_dir = os.path.join("D:/Projects/mmocr/output", dt_string)
    run_det_exp(model_name_list=model_name_list, dir_list=dir_list, exp_dir=exp_dir)


if __name__ == "__main__":
    main_det()
