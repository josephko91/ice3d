# import packages
import numpy as np
import cadquery as cq
from scipy.stats import qmc
import math
import itertools
import matplotlib.pyplot as plt
import os, re
import json
import copy
import random
import sys
import miniball
import pandas as pd
import glob
import contextlib

# ========= Functions ===========

### =========== (Rosette code) =========== ###
def create_bullet(a, c, hp, f_a, f_c, workplane):
    # create pyramid
    n_pyr = 6
    ri = a*np.cos(np.radians(30)) # distance between center and edge of hexagon
    theta = 90 - np.degrees(np.arctan(hp/ri))
    pyramid = workplane.polygon(n_pyr, f_a*2*a).extrude(-f_a*hp, taper=theta)
    # create cylinder 
    n_cyl = 6
    cylinder = workplane.polygon(n_cyl, f_a*2*a).extrude(f_c*2*c)
    # create bullet (union)
    bullet = cylinder.union(pyramid)
    return bullet

def calc_r0(f_r0, a, n_arms):
    '''
    linearly interpolate between perscribed limits for r0
    '''
    ymin, ymax = 0.5*a, 1*a
    xmin, xmax = 4, 12
    slope = (ymax-ymin)/(xmax-xmin)
    intercept = ymin - (slope*xmin)
    r0 = slope*(n_arms) + intercept
    r0 = f_r0 * r0 # multiply by perturbation factor
    return r0 

def calc_hp(f_hp, r0, n_arms):
    '''
    linearly interpolate: hp increases with n_arms
    '''
    ymin, ymax = 1*r0, 1.5*r0
    xmin, xmax = 4, 12
    slope = (ymax-ymin)/(xmax-xmin)
    intercept = ymin - (slope*xmin)
    hp = slope*(n_arms) + intercept
    hp = f_hp*hp # multiply by perturbation factot
    return hp

def calc_h0(f_h0, r0):
    '''
    h0 calculate as a perscribed fraction of r0
    '''
    h0 = r0/2
    h0 = f_h0*h0 # multiply by perturbation factor
    return h0

def extract_xyz(s_code):
    '''
    Convert list in format [x1, y1, z1, ..., xn, yn, zn] to separate x, y, z arrays
    '''
    x = []
    y = []
    z = []
    for i in range(0, len(s_code), 3):
        x.append(s_code[i])
        y.append(s_code[i+1])
        z.append(s_code[i+2])
    return x, y, z

# create_ros(base_params, n_arms, s_code, aspect_perturb)
def create_ros(params, n_arms, s_code, aspect_perturb):
    '''
    aspect_perturb: list in form [f_a_1,f_c_1,...,f_a_n_arms,f_c_n_arms]
    '''
    # unpack parameters
    a, c, f_r0, f_hp, f_h0 = params[0], params[1], params[2], params[3], params[4]
    r0 = calc_r0(f_r0, a, n_arms)
    hp = calc_hp(f_hp, r0, n_arms)
    h0 = calc_h0(f_h0, r0)
    # create sphere
    sphere = cq.Workplane().sphere(r0)
    # create outer shell to "place" bullets on
    # based on spherical code from Sloane et al. 
    r_outer = r0 + hp - h0
    # convert s_code list to outer_coords
    x, y, z = extract_xyz(s_code)
    outer_coords = r_outer*(np.column_stack((x, y, z)))
    # create and collect bullets in list
    bullets = []
    for i in range(len(outer_coords)):
        f_a = aspect_perturb[2*i]
        f_c = aspect_perturb[2*i + 1]
        normal_vector = tuple(outer_coords[i])
        plane = cq.Plane(origin=normal_vector, normal=normal_vector)
        workplane = cq.Workplane(plane)
        bullet = create_bullet(a, c, hp, f_a, f_c, workplane)
        bullets.append(bullet)
    # boolean union to create rosette
    ros = sphere.union(bullets[0])
    for i in range(1, n_arms):
        ros = ros.union(bullets[i])
    return ros
### ====================================== ###
def get_verts(ros, threshold):
    verts = ros.vertices() # list of vertices 
    origin = cq.Vertex.makeVertex(0,0,0)
    # filtered_verts = [v for v in verts if v.distance(origin) > threshold/2]
    filtered_verts = [v for v in verts]
    final_verts = np.asarray([list(v.Center().toTuple()) for v in filtered_verts])
    return final_verts 
    
def calc_mbs(points):
    """
    Calculate minimal bounding sphere (mbs)
    """
    mbs = {} # store attributes of sphere as dict

    # use miniball algorithm to find bounding sphere
    # mesh_points = np.asarray(points)
    unique_pts = np.unique(points, axis=0)
    c, r2 = miniball.get_bounding_ball(unique_pts)
    r = np.sqrt(r2) # r2 = radius squared, r = radius

    mbs['c'] = c # center coordinates as np array
    mbs['r'] = r # radius of sphere as float
    mbs['v'] = (4/3)*np.pi*(r**3)
    mbs['a'] = 4*np.pi*(r**2) 

    return mbs

def get_record(ros, params, id):
    base_params = params[0]
    record = [id]
    record.extend(base_params)
    if ros==None:
        # -666 is tag for ros creation bug
        sa = vol = sa_eff = rho_eff = -666
    else:
        try:
            sa = ros.val().Area()
            vol = ros.val().Volume()
            points = get_verts(ros, base_params[2])
            mbs = calc_mbs(points)
            rho_eff = vol/mbs['v'] 
            sa_eff = sa/mbs['a']
        except Exception as e:
            # -999 is tag for mbs bug
            sa = vol = sa_eff = rho_eff = -999
            print(f'Rosette {id} mbs_error: {e}')
    record.extend([sa, vol, sa_eff, rho_eff])
    return record

def process_instance(params, i, save_dir, task_index):
    # extract params
    base_params = params[0][:5]
    n_arms = params[0][5]
    aspect_perturb = params[1]
    s_code = params[2]
    # make stl and record dirs if they don't exist
    record_dir = save_dir + f'/data'
    stl_dir = save_dir + f'/stl/{n_arms}'
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(stl_dir, exist_ok=True)
    try: 
        ros = create_ros(base_params, n_arms, s_code, aspect_perturb)
        save_filename = f'ros-test-{i:06d}.stl'
        save_filepath = os.path.join(stl_dir, save_filename)
        cq.exporters.export(ros, save_filepath) # save file
    except Exception as e:
        ros = None
        print(f'create_ros error: {e}')
    # calc attributes and save record as txt
    record = get_record(ros, params, i)
    record_filename = f'ros-data-{task_index}.txt'
    record_filepath = os.path.join(record_dir, record_filename)
    # print(record_filepath)
    with open(record_filepath, 'a') as file:
        file.write(",".join(map(str, record)) + '\n') 
    
def process_chunk(chunk, start_index, end_index, save_dir, task_index):
    for i in range(start_index, end_index):
        p = chunk[i]
        process_instance(p, i, save_dir, task_index)

# =========== Main ================
def main():
    # set directories
    save_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/residual2'
    os.makedirs(save_dir, exist_ok=True)
    # load the param file that was used to create rosettes
    params_path = '/glade/u/home/joko/ice3d/output/params_200_50.json'
    # load saved json
    with open(params_path, 'rb') as file:
        params_list = json.load(file)
    # load the generated rosette data
    data_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/data/'
    file_paths = glob.glob(data_dir + "*.txt")
    dfs = []
    for file in file_paths:
        df = pd.read_csv(file, delimiter=",", header=None)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    colnames = ['id', 'a', 'c', 'f_r0', 'f_hp', 'f_h0', 'n_arms', 'sa', 'vol', 'sa_eff', 'rho_eff']
    df.columns = colnames
    # ------ Identify missing rosettes ------ #
    stl_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/stl/'
    # Get all .stl files, including those in subdirectories
    file_paths = glob.glob(stl_dir + '**/*.stl', recursive=True)
    # Initialize an empty list to store the gen_ids
    gen_ids = []
    # Regular expression to extract the number
    pattern = r'ros-test-(\d+)\.stl'
    # Loop through the files and extract the number
    for file in file_paths:
        match = re.search(pattern, file)
        if match:
            # Extract the number and append it to the list
            number = match.group(1)
            gen_ids.append(int(number))  # Convert to integer if needed
    # get missing ids and put in list
    full_ids = list(range(70000))
    missing = list(set(full_ids) - set(gen_ids))
    # ------ Identify rosettes with mbs error ------ #
    mbs_err = df[df['sa']==-999]
    mbs_list = list(mbs_err['id'].values)
    union = list(set(mbs_list) | set(missing)) # combine both lists
    # ------ Re-create rosettes and do calculations ------ #    
    for i in range(len(union)):
        id = union[i]
        print(f'Processing ros id {id}')
        params = params_list[id] # set here
        base_params = params[0][:5]
        n_arms = params[0][5]
        aspect_perturb = params[1]
        s_code = params[2]
        # reduce sigfigs
        base_params = [round(i, 2) for i in base_params]
        s_code = [round(i, 2) for i in s_code]
        aspect_perturb = [round(i, 2) for i in aspect_perturb]
        # make stl and record dirs if they don't exist
        record_dir = save_dir + '/data'
        stl_dir = save_dir + f'/stl/{n_arms}'
        os.makedirs(record_dir, exist_ok=True)
        os.makedirs(stl_dir, exist_ok=True)
        try: 
            ros = create_ros(base_params, n_arms, s_code, aspect_perturb) # create ros object
            save_filename = f'ros-test-{id:06d}.stl'
            save_filepath = os.path.join(stl_dir, save_filename)
            print(f'saving to {save_filepath}')
            cq.exporters.export(ros, save_filepath) # save 
            # create and save data record
            record = get_record(ros, params, id)
            record_filename = 'ros-data-res.txt'
            record_filepath = os.path.join(record_dir, record_filename)
            with open(record_filepath, 'a') as file:
                file.write(",".join(map(str, record)) + '\n') 
            if os.path.exists(save_filepath):
                print(f'--> Rosette {id} re-generated successfully!')
            else:
                print(f'Rosette {id} was generated but not saved properly')
        except Exception as e:
            print(f'Rosette {id} not created, error: {e}')

if __name__ == "__main__":
    main()

# to run use: python 05-gen-ros-residuals.py > output_04.log 2> error_04.log
