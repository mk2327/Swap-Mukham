import os
import cv2
import glob
import time
import torch
import shutil
import argparse
import platform
import datetime
import subprocess
import insightface
import onnxruntime
import numpy as np
import gradio as gr
import threading
import queue
from tqdm import tqdm
import concurrent.futures

# Define the scale_bbox_from_center function
def scale_bbox_from_center(bbox, scale_width, scale_height, image_width, image_height):
    # Extract the coordinates of the bbox
    x1, y1, x2, y2 = bbox

    # Calculate the center point of the bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the new width and height of the bbox based on the scaling factors
    width = x2 - x1
    height = y2 - y1
    new_width = width * scale_width
    new_height = height * scale_height

    # Calculate the new coordinates of the bbox, considering the image boundaries
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Adjust the coordinates to ensure the bbox remains within the image boundaries
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width - 1, new_x2)
    new_y2 = min(image_height - 1, new_y2)

    # Return the scaled bbox coordinates
    scaled_bbox = [new_x1, new_y1, new_x2, new_y2]
    return scaled_bbox

#face analyser
swap_options_list = ["All Face"]

def get_single_face(faces):
    return sorted(faces, key=lambda face: face["det_score"])[-1]

def analyse_face(image, model, return_single_face=True, detect_condition="best detection", scale=1.0):
    faces = model.get(image)
    
    if scale != 1:
        for i, face in enumerate(faces):
            landmark = face['kps']
            center = np.mean(landmark, axis=0)
            landmark = center + (landmark - center) * scale
            faces[i]['kps'] = landmark

    if not return_single_face:
        return faces

    return get_single_face(faces)

def get_analysed_data(face_analyser, image_sequence, source_data, swap_condition="All Face", scale=1.0):
    source_path, age = source_data
    source_image = cv2.imread(source_path)
    analysed_source = analyse_face(source_image, face_analyser, return_single_face=True, scale=scale)

    analysed_target_list = []
    analysed_source_list = []
    whole_frame_eql_list = []
    num_faces_per_frame = []

    for frame_path in tqdm(image_sequence, desc="Analysing face data"):
        frame = cv2.imread(frame_path)
        analysed_faces = analyse_face(frame, face_analyser, return_single_face=False, scale=scale)

        for analysed_face in analysed_faces:
            if swap_condition == "All Face":
                analysed_target_list.append(analysed_face)
                analysed_source_list.append(analysed_source)
                whole_frame_eql_list.append(frame_path)

        num_faces_per_frame.append(len(analysed_faces))

    return analysed_target_list, analysed_source_list, whole_frame_eql_list, num_faces_per_frame

#

#main process
detect_conditions = ["best detection"]

def get_single_face(faces, method="best detection"):
    total_faces = len(faces)
    if total_faces == 1:
        return faces[0]

    print(f"{total_faces} face detected. Using {method} face.")
    if method == "best detection":
        return sorted(faces, key=lambda face: face["det_score"])[-1]

def analyse_face(image, model, return_single_face=True, detect_condition="best detection", scale=1.0):
    faces = model.get(image)
    if scale != 1:
        for i, face in enumerate(faces):
            landmark = face['kps']
            center = np.mean(landmark, axis=0)
            landmark = center + (landmark - center) * scale
            faces[i]['kps'] = landmark

    if not return_single_face:
        return faces

    return get_single_face(faces, method=detect_condition)

def get_analysed_data(face_analyser, image_sequence, source_data, swap_condition="All face", detect_condition="best detection", scale=1.0):
    if swap_condition != "Specific Face":
        source_path, age = source_data
        source_image = cv2.imread(source_path)
        analysed_source = analyse_face(source_image, face_analyser, return_single_face=True, detect_condition=detect_condition, scale=scale)
    else:
        analysed_source_specifics = []
        source_specifics, threshold = source_data
        for source, specific in zip(*source_specifics):
            if source is None or specific is None:
                continue
            analysed_source = analyse_face(source, face_analyser, return_single_face=True, detect_condition=detect_condition, scale=scale)
            analysed_specific = analyse_face(specific, face_analyser, return_single_face=True, detect_condition=detect_condition, scale=scale)
            analysed_source_specifics.append([analysed_source, analysed_specific])

    analysed_target_list = []
    analysed_source_list = []
    whole_frame_eql_list = []
    num_faces_per_frame = []

    total_frames = len(image_sequence)
    for frame_path in tqdm(image_sequence, total=total_frames, desc="Analysing face data"):
        frame = cv2.imread(frame_path)
        analysed_faces = analyse_face(frame, face_analyser, return_single_face=False, detect_condition=detect_condition, scale=scale)

        n_faces = 0
        for analysed_face in analysed_faces:
            if swap_condition == "All Face":
                analysed_target_list.append(analysed_face)
                analysed_source_list.append(analysed_source)
                whole_frame_eql_list.append(frame_path)
                n_faces += 1

        num_faces_per_frame.append(n_faces)

    return analysed_target_list, analysed_source_list, whole_frame_eql_list, num_faces_per_frame

def process_images(input_folder):
    global WORKSPACE
    global OUTPUT_FILE
    global PREVIEW
    WORKSPACE, OUTPUT_FILE, PREVIEW = None, None, None

    source_path = "source_image.jpg"  # Provide the path to your source image
    age = 30
    distance = 0.6
    swap_condition = "All Face"
    enable_face_parser = True
    includes = ["face"]
    mask_soft_kernel = 5
    mask_soft_iterations = 2
    blur_amount = 0
    erode_amount = 0
    face_scale = 1.0
    enable_laplacian_blend = False
    crop_top = 0
    crop_bott = 0
    crop_left = 0
    crop_right = 0
    specifics = []

    image_sequence = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.ico', '.webp'))]

    for image_path in image_sequence:
        try:
            analysed_targets, analysed_sources, whole_frame_list, num_faces_per_frame = get_analysed_data(
                FACE_ANALYSER,
                [image_path],
                source_path, age if swap_condition != "Specific Face" else ((sources, specifics), distance),
                swap_condition=swap_condition,
                detect_condition="best detection",
                scale=face_scale
            )

            # Rest of the code to process the current image
            # ...

        except Exception as e:
            print(f"Error processing image '{image_path}': {e}")
            continue

if __name__ == "__main__":
    input_folder = 'inputs'  # Replace 'inputs' with the actual folder name
    process_images(input_folder)
