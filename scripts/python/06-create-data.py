'''
This script calculates:
(1) inputs/outputs necessary for supervised tabular ML 
(2) simultaneously, this dataframe can be used for deep learning training
- i.e., ID / output pairs 
'''
import cv2 as cv
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import numpy as np

### ========= Helper functions ========= ###

def get_img(file_path):
    img = cv.imread(file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)[...,0]
    return img

def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha, cmap=plt.cm.gray)
    ax.set_axis_off()
    return ax

def get_border(image, width):
    bg = np.zeros(image.shape)
    contours, _ = cv.findContours(image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    biggest = 0
    bigcontour = None
    for contour in contours:
        area = cv.contourArea(contour) 
        if area > biggest:
            biggest = area
            bigcontour = contour
    return cv.drawContours(bg, [bigcontour], 0, (255, 255, 255), width).astype(bool), contours 

def get_aspect_ratio(cnt):
    rect = cv.minAreaRect(cnt)
    # get length and width of contour
    x = rect[1][0]
    y = rect[1][1]
    rect_length = max(x, y)
    rect_width = min(x, y)
    phi = rect_width / rect_length
    return phi

def get_aspect_ratio_elip(cnt):
    ellipse = cv.fitEllipse(cnt)
    # Get width and height of rotated ellipse
    widthE = ellipse[1][0]
    heightE = ellipse[1][1]
    if widthE > heightE:
        phiE = heightE / widthE
    else:
        phiE = widthE / heightE
    return phiE

def get_extreme_pts(cnt):
    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    top = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])
    extreme_pts = np.std([left, right, top, bottom])
    return extreme_pts

def get_contour_area(cnt):
    area = cv.contourArea(cnt)
    return area

def get_contour_perimeter(cnt):
    perimeter = cv.arcLength(cnt, True)
    return perimeter

def get_min_circle(cnt):
    center ,radius = cv.minEnclosingCircle(cnt)
    perimeter_circle = 2*np.pi*radius
    area_circle = np.pi*(radius**2)
    return center, radius, perimeter_circle, area_circle

def get_area_ratio(cnt):
    area = get_contour_area(cnt)
    _,_,_,area_circle = get_min_circle(cnt)
    area_ratio = area/area_circle
    return area_ratio

def get_complexity(cnt):
    _, radius, _, _ = get_min_circle(cnt)
    area = get_contour_area(cnt)
    perimeter = get_contour_perimeter(cnt)
    Ac = np.pi * radius ** 2
    complexity = 10*(0.1-(area / (np.sqrt(area / Ac) * perimeter ** 2)))
    return complexity

def get_circularity(cnt):
    area = get_contour_area(cnt)
    perimeter = get_contour_perimeter(cnt)
    circularity = 4*np.pi*(area/(perimeter**2))
    return circularity

def process_img(path):
    img = get_img(path)
    _, contours = get_border(img, 5)
    cnt = contours[0]
    aspect_ratio = get_aspect_ratio(cnt)
    aspect_ratio_elip = get_aspect_ratio_elip(cnt)
    extreme_pts = get_extreme_pts(cnt)
    contour_area = get_contour_area(cnt)
    contour_perimeter = get_contour_perimeter(cnt)
    area_ratio = get_area_ratio(cnt)
    complexity = get_complexity(cnt)
    circularity = get_circularity(cnt)
    img_features = [aspect_ratio, aspect_ratio_elip, extreme_pts, \
        contour_area, contour_perimeter, area_ratio, \
        complexity, circularity]
    return img_features

def process_instance(img_path, save_dir, df_ros, task_index):
    # initiate data record 
    data_record = []
    ros_id = img_path.rsplit('-',2)[1]
    proj_id = img_path.rsplit('-',2)[2].split('.',2)[0]
    unique_id = ros_id + '_' + proj_id
    params_output = list(df_ros[df_ros['id']==int(ros_id)].iloc[0]) # inputs params + outputs
    data_record.extend([unique_id, ros_id, proj_id])
    data_record.extend(params_output[1:])
    # open image
    img = get_img(img_path)
    # run processing functions on img
    _, contours = get_border(img, 5)
    cnt = contours[0]G
    aspect_ratio = get_aspect_ratio(cnt)
    aspect_ratio_elip = get_aspect_ratio_elip(cnt)
    extreme_pts = get_extreme_pts(cnt)
    contour_area = get_contour_area(cnt)
    contour_perimeter = get_contour_perimeter(cnt)
    area_ratio = get_area_ratio(cnt)
    complexity = get_complexity(cnt)
    circularity = get_circularity(cnt)
    img_features = [aspect_ratio, aspect_ratio_elip, extreme_pts, \
        contour_area, contour_perimeter, area_ratio, \
        complexity, circularity]
    # create data record for this img instance 
    data_record.extend(img_features) 
    # append to data file (one file per core to split up data)
    record_filename = f'ros-tabular-data-{task_index}.txt'
    record_dir = os.path.join(save_dir, 'tabular-data')
    os.makedirs(record_dir, exist_ok=True) # make dir if doesn't exist
    record_filepath = os.path.join(record_dir, record_filename)
    # test prints
    print(f'writing data to -> {record_filepath}')
    print(f'data -> {data_record}')
    with open(record_filepath, 'a') as file: 
        file.write(",".join(map(str, data_record)) + '\n') 

def process_chunk(img_paths, start_index, end_index, save_dir, df_ros,  task_index):
    for i in range(start_index, end_index):
        img_path = img_paths[i]
        process_instance(img_path, save_dir, df_ros, task_index)

### ========= Main ========= ###
def main():
    save_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316'
    projections_dir = os.path.join(save_dir, 'projections')
    ros_data = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/data/ros-data-merged.txt'
    df_ros = pd.read_csv(ros_data) # data: params + outputs
    # get list of paths of all images
    img_paths_txt = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/projections/img_relative_paths.txt'
    with open(img_paths_txt, 'r') as file:
        rel_paths = [line.strip().replace('./','') for line in file]
        img_paths = [os.path.join(projections_dir, i) for i in rel_paths]
    # img_paths = img_paths[:1000] # for testing 
    print(f'Creating data for {len(img_paths)} png files!!!')
    # Get the total number of tasks (this will be passed by PBS)
    num_tasks = int(sys.argv[1])
    # Get the job index from PBS (PBS_ARRAYID is the index for each job in the array)
    task_index = int(sys.argv[2])
    # Calculate the chunk size for each task
    chunk_size = len(img_paths) // num_tasks  # Integer division
    remainder = len(img_paths) % num_tasks
    # Calculate the start and end indices for the current task
    start_index = task_index * chunk_size
    if (task_index == (num_tasks-1)) & (remainder > 0):
        end_index = start_index + chunk_size + remainder 
    else:
        end_index = start_index + chunk_size
    # check if in index, process data chunk
    if start_index < len(img_paths):
        print(f'processing chunk {start_index}:{end_index}')
        process_chunk(img_paths, start_index, end_index, save_dir, df_ros, task_index)
    else:
        sys.exit() # out of index
    
if __name__ == "__main__":
    main()