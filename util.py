import os
import json
import zipfile
import shutil
import cv2
import random

from mmocr.models.textdet.dense_heads import db_head, drrg_head, fce_head, pan_head, pse_head
from mmocr.models.textdet.dense_heads.drrg_head import DRRGHead

def unzip_videos(folder):
    zipped_file_list = [f for f in os.listdir(folder) if os.path.splitext(f)[1] == ".zip"]
    for zipped_file in zipped_file_list:
        with zipfile.ZipFile(os.path.join(folder, zipped_file), 'r') as zip_ref:
            zip_ref.extractall(folder)


def task_1():
    """
    Task 1: Unzip videos in the training set
    """
    folder = "D:/Data/Text/BOVText/training/Video"
    unzip_videos(folder)

def containOnlyOverlay(ann_dict):
    """
    Check if the video only contains overlay text according to the annotation
    """
    for frameID in ann_dict:
        if ann_dict[frameID]:
            for ann in ann_dict[frameID]:
                if not (ann["category"] == "title" or ann["category"] == "caption"):
                    return False
    return True

def extract_overlay_subset_video(src_folder, tgt_folder):
    """
    Select the video that ONLY contains overlay text, according to the annotation
    """
    split_list = ["training", "test"]
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
    subfolder_list = [subfolder for subfolder in split_list]

    total_video_num = 0
    overlay_video_num = 0
    for subfolder in subfolder_list:
        scene_folder_list = [s for s in os.listdir(os.path.join(src_folder, subfolder, "Annotation"))]
        for scene_folder in scene_folder_list:
            ann_file_list = [f for f in os.listdir(os.path.join(src_folder, subfolder,"Annotation",scene_folder)) if os.path.splitext(f)[1] == ".json"]
            for anno_file in ann_file_list:
                src_ann_file = os.path.join(src_folder, subfolder, "Annotation", scene_folder, anno_file)
                with open(src_ann_file, encoding="utf8", mode='r') as a:
                    anno_dict = json.load(a)
                    total_video_num += 1
                    if (containOnlyOverlay(anno_dict)):
                        overlay_video_num += 1
                        # copy the annotation file
                        tgt_ann_folder = os.path.join(tgt_folder, subfolder, "Annotation", scene_folder)
                        if not os.path.exists(tgt_ann_folder):
                            os.makedirs(tgt_ann_folder)
                        tgt_ann_file = os.path.join(tgt_ann_folder, anno_file)
                        shutil.copyfile(src_ann_file,tgt_ann_file)
                        # copy the video
                        tgt_vid_folder = os.path.join(tgt_folder, subfolder, "Video", scene_folder)
                        if not os.path.exists(tgt_vid_folder):
                            os.makedirs(tgt_vid_folder)
                        vid_file = os.path.splitext(anno_file)[0]+".mp4"
                        src_vid_file = os.path.join(src_folder, subfolder, "Video", scene_folder, vid_file)
                        tgt_vid_file = os.path.join(tgt_vid_folder, vid_file)
                        shutil.copyfile(src_vid_file,tgt_vid_file)
    print("Total : "+str(total_video_num))
    print("Overlay : "+str(overlay_video_num))


def task_2():
    """
    Task 2: Extract the overlay video subset of the BOVText datatset
    """
    src_folder = "D:/Data/Text/BOVText"
    tgt_folder = "D:/Data/Text/BOVText_Overlay_Video"
    extract_overlay_subset_video(src_folder=src_folder, tgt_folder=tgt_folder)


def process_videos(src_folder, tgt_folder, split_folder_list):
    """
    Convert the BOV video to images following icdar format
    """
    tgt_img_folder = os.path.join(tgt_folder, "imgs")

    # process videos to generate imgs
    for split_folder in split_folder_list:
        sub_folder = os.path.join(src_folder, split_folder)
        video_folder = os.path.join(sub_folder, "Video")
        scene_list = os.listdir(video_folder)
        for scene_name in scene_list:
            scene_folder = os.path.join(video_folder, scene_name)
            video_list = os.listdir(scene_folder)
            for video_name in video_list:
                video_path = os.path.join(scene_folder, video_name)
                video = cv2.VideoCapture(video_path)
                frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
                print(f"Writing {video_name}, it has {frame_count} frames")
                frame_idx = 1 # begin with 1
                while video.isOpened():
                    ret, frame = video.read()
                    if ret:
                        output_frame_name = f"{os.path.splitext(video_name)[0]}_{frame_idx:06d}.png"
                        output_frame_path = os.path.join(tgt_img_folder, split_folder, output_frame_name)
                        cv2.imwrite(output_frame_path, frame)
                        frame_idx += 1


def process_videos_and_anns(src_folder, tgt_folder, split_folder_list):
    """
    Convert the BOV annotation to icdar format, and the video to images.
    To save space, only save frames that have text
    """
    tgt_img_folder = os.path.join(tgt_folder, "imgs")
    tgt_ann_folder = os.path.join(tgt_folder, "annotations")

    total_ann_frame = 0
    for split_folder in split_folder_list:
        instance_file = f"instances_{split_folder}.json"
        instances = {"images": [], "categories": [{"id": 1, "name": "text"}], "annotations": []}
        sub_folder = os.path.join(src_folder, split_folder)
        anno_folder = os.path.join(sub_folder, "Annotation")
        video_folder = os.path.join(sub_folder, "Video")
        scene_list = os.listdir(video_folder)
        instance_img_id = 0
        instance_ann_id = 0
        for scene_name in scene_list:
            scene_folder = os.path.join(video_folder, scene_name)
            video_list = os.listdir(scene_folder)
            for video_name in video_list:
                # Load annotation
                anno_path = os.path.join(anno_folder, scene_name, os.path.splitext(video_name)[0]+".json")
                with open(anno_path, encoding="utf8", mode='r') as a:
                    anno_dict = json.load(a)
                # Load video
                video_path = os.path.join(scene_folder, video_name)
                video = cv2.VideoCapture(video_path)
                frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
                print(f"Writing {video_name}, it has {frame_count} frames")
                frame_idx = 1  # begin with 1
                has_anno_num = 0
                while video.isOpened():
                    ret, frame = video.read()
                    if ret:
                        if f"{frame_idx}" not in anno_dict:
                            print(f"{video_name} does not have frame {frame_idx}")
                            continue
                        if len(anno_dict[f"{frame_idx}"]) != 0:
                            has_anno_num += 1
                            # save the image
                            output_frame_name = f"{os.path.splitext(video_name)[0]}_{frame_idx:06d}.png"
                            output_frame_path = os.path.join(tgt_img_folder, split_folder, output_frame_name)
                            cv2.imwrite(output_frame_path, frame)
                            # save the annotation
                            output_ann_name = f"{os.path.splitext(video_name)[0]}_{frame_idx:06d}.txt"
                            output_ann_path = os.path.join(tgt_ann_folder, split_folder, output_ann_name)
                            # instance json
                            img_info = {}
                            img_info["file_name"] = f"{split_folder}/{output_frame_name}"
                            img_info["height"] = frame.shape[0]
                            img_info["width"] = frame.shape[1]
                            img_info["segm_file"] = f"{split_folder}/{output_ann_name}"
                            img_info["id"] = instance_img_id
                            instance_img_id += 1

                            instances["images"].append(img_info)
                            with open(output_ann_path, 'w', encoding='utf-8') as f:
                                for gt in anno_dict[f"{frame_idx}"]:
                                    ann_info = {}
                                    ann_info["iscrowd"] = 0
                                    ann_info["category_id"] = 1
                                    bbox = gt["points"]
                                    x1 = int(float(bbox[0]))
                                    y1 = int(float(bbox[1]))
                                    x2 = int(float(bbox[2]))
                                    y2 = int(float(bbox[3]))
                                    x3 = int(float(bbox[4]))
                                    y3 = int(float(bbox[5]))
                                    x4 = int(float(bbox[6]))
                                    y4 = int(float(bbox[7]))
                                    ann_info["bbox"] = [min(x1, x4), min(y1, y2), max(x2, x3)-min(x1, x4), max(y3, y4)-min(y1, y2)]
                                    ann_info["area"] = (max(x2, x3)-min(x1, x4))*(max(y3, y4)-min(y1, y2))
                                    ann_info["segmentation"] = [[x1, y1, x2, y2, x3, y3, x4, y4]]
                                    ann_info["image_id"] = instance_img_id
                                    ann_info["id"] = instance_ann_id
                                    instances["annotations"].append(ann_info)
                                    instance_ann_id += 1
                                    line = ""
                                    for coord in bbox:
                                        line += str(int(float(coord)))+","
                                    line += gt["transcription"]+"\n"
                                f.write(line)

                        frame_idx += 1
                    else:
                        break
                total_ann_frame += has_anno_num
                print(f"{video_name} has {frame_count} frames in total, {has_anno_num} has text annotation")
        # write instance_file
        with open(instance_file, 'w') as f:
            json.dump(instances, f)
    print(f"{total_ann_frame} frames generated")


def task_3():
    """
    Covert BOVOverlay to ICDAR style
    """
    src_folder = "D:/Data/Text/BOVText_Overlay_Video"
    tgt_folder = "D:/Projects/mmocr/data/bovoverlay"

    split_folder_list = ["training", "test"]
    process_videos_and_anns(src_folder=src_folder, tgt_folder=tgt_folder, split_folder_list=split_folder_list)


def task_4():
    """
    Shuffle the dataset
    """
    src_json_file_list = ["data/bovoverlay/instances_test.json",
                          "data/bovoverlay/instances_training.json"]
    tgt_json_file_list = ["data/bovoverlay/instances_test_shuffled.json",
                          "data/bovoverlay/instances_training_shuffled.json"]

    for i in range(len(src_json_file_list)):
        src_json_file = src_json_file_list[i]
        tgt_json_file = tgt_json_file_list[i]
        # Load and shuflle
        with open(src_json_file, 'r', encoding='utf-8') as f:
            anno = json.load(f)
            random.shuffle(anno["images"])
            random.shuffle(anno["annotations"])
        # Save
        with open(tgt_json_file, 'w', encoding='utf-8') as f:
            json.dump(anno, f)


def task_5():
    """
    Select ~100 images from test
    """
    src_dir = "D:/Projects/mmocr/data/bovoverlay/imgs/test"
    tgt_dir = "D:/Projects/mmocr/data/dms/overlay_test"
    step = 460

    img_file_list = os.listdir(src_dir)
    for i in range(0,len(img_file_list), step):
        src_file = os.path.join(src_dir,img_file_list[i])
        tgt_file = os.path.join(tgt_dir,img_file_list[i])
        shutil.copyfile(src_file, tgt_file)
    print("Done")


def task_6():
    """
    Check heads size
    """
    # DBNet
    dbHead = db_head.DBHead(in_channels=256)
    num_parameters = sum(p.numel() for p in dbHead.parameters() if p.requires_grad)
    print(f"DBNet head has {num_parameters} parameters")

    # DRRG
    drrgHead =  drrg_head.DRRGHead(in_channels=32)
    num_parameters = sum(p.numel() for p in drrgHead.parameters() if p.requires_grad)
    print(f"DRRG head has {num_parameters} parameters")

    # FCE
    fceHead = fce_head.FCEHead(in_channels=256, scales=(8, 16, 32))
    num_parameters = sum(p.numel() for p in fceHead.parameters() if p.requires_grad)
    print(f"FCE head has {num_parameters} parameters")

    # PAN
    panHead = pan_head.PANHead(in_channels=[128, 128, 128, 128],out_channels=6)
    num_parameters = sum(p.numel() for p in panHead.parameters() if p.requires_grad)
    print(f"PAN head has {num_parameters} parameters")

    # PSE
    pseHead = pse_head.PSEHead(in_channels=[256], out_channels=7)
    num_parameters = sum(p.numel() for p in pseHead.parameters() if p.requires_grad)
    print(f"PSE head has {num_parameters} parameters")


def main():
    #task_3()
    #task_4()
    #task_5()
    task_6()


if __name__ == "__main__":
    main()