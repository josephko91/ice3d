# import packages
import numpy as np
import cadquery as cq
from scipy.stats import qmc
import math
import itertools
import matplotlib.pyplot as plt
import os
import json
import copy
import random
import sys
import miniball

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
    try:
        sa = ros.val().Area()
        vol = ros.val().Volume()
        base_params = params[0]
        # print(f'rosette {id}: {base_params}')
        points = get_verts(ros, base_params[2])
        mbs = calc_mbs(points)
        rho_eff = vol/mbs['v'] 
        sa_eff = sa/mbs['a']
        record = [id]
        record.extend(base_params)
        record.extend([sa, vol, sa_eff, rho_eff])
        return record
    except Exception as e:
        print(f'rosette {id}: {base_params}')
        return f"An unexpected error occurred: {e}"

def process_instance(params, i, save_dir):
    # extract params
    base_params = params[0][:5]
    n_arms = params[0][5]
    aspect_perturb = params[1]
    s_code = params[2]
    ros = create_ros(base_params, n_arms, s_code, aspect_perturb)
    # make stl and record dirs if they don't exist
    record_dir = save_dir + f'/data/{n_arms}'
    stl_dir = save_dir + f'/stl/{n_arms}'
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(stl_dir, exist_ok=True)
    # calc attributes and save record as txt
    record = get_record(ros, params, i)
    record_filename = f'record-ros-test-{i:06d}.txt'
    record_filepath = os.path.join(record_dir, record_filename)
    # print(record_filepath)
    with open(record_filepath, 'w') as file:
        file.write(",".join(map(str, record))) 
    # save model
    save_filename = f'ros-test-{i:06d}.stl'
    save_filepath = os.path.join(stl_dir, save_filename)
    # print(save_filepath)
    # print(type(ros))
    cq.exporters.export(ros, save_filepath) # save file

def process_chunk(chunk, start_index, end_index, save_dir):
    for i in range(start_index, end_index):
        p = chunk[i]
        process_instance(p, i, save_dir)

def main():
    # set directory to save data
    save_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50-20250314'
    os.makedirs(save_dir, exist_ok=True)
    # Load the JSON file
    params_path = '/glade/u/home/joko/ice3d/output/params_200_50.json'
    with open(params_path, 'rb') as file:
        params = json.load(file)

    # Get the total number of tasks (this will be passed by PBS)
    num_tasks = int(sys.argv[1])
    
    # Get the job index from PBS (PBS_ARRAYID is the index for each job in the array)
    task_index = int(sys.argv[2])
    
    # Calculate the chunk size for each task
    chunk_size = len(params) // num_tasks  # Integer division
    remainder = len(params) % num_tasks
    
    # Calculate the start and end indices for the current task
    start_index = task_index * chunk_size
    if (task_index == (num_tasks-1)) & (remainder > 0):
        end_index = start_index + chunk_size + remainder 
    else:
        end_index = start_index + chunk_size 

    # check if in index, process data chunk
    if start_index < len(params):
        # # Get the chunk for this task
        # chunk = params[start_index:end_index]
        # Process the chunk
        print(f'processing chunk {start_index}:{end_index}')
        process_chunk(params, start_index, end_index, save_dir)
    else:
        sys.exit() # out of index

if __name__ == "__main__":
    main()

