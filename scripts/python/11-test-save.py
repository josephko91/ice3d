import multiprocessing
import pyvista as pv
import os, sys
import random
import numpy as np
import json
import time
from functools import partial

def init_plotter(res, bg_color):
    pl = pv.Plotter(off_screen=True, window_size=[res, res])
    pl.background_color = bg_color
    pl.enable_parallel_projection()
    pl.remove_all_lights()
    return pl 
    
def random_rotate(mesh):
    rotated = mesh.copy()
    deg_x = np.random.randint(1, 360)
    deg_y = np.random.randint(1, 360)
    deg_z = np.random.randint(1, 360)
    rotated.rotate_x(deg_x, inplace=True)
    rotated.rotate_y(deg_y, inplace=True)
    rotated.rotate_z(deg_z, inplace=True)
    return rotated
    
def get_orthogonal_vector(v):
    '''
    Return an arbitrary orthogonal vector to v
    '''    
    arbitrary_vector = np.array([1, 0, 0])
    v_orth = np.cross(v, arbitrary_vector)
    # If v1 and arbitrary_vector are parallel, you could end up with a zero vector, 
    # in which case you'd need to choose a different arbitrary vector. Let's check:
    if np.all(v_orth == 0):
        arbitrary_vector = np.array([0, 1, 0]) # set to another arbitrary vector (e.g., [0, 1, 0])
        v_orth = np.cross(v1, arbitrary_vector)
    # Normalize the orthogonal vector v2
    v_orth = v_orth / np.linalg.norm(v_orth)
    return v_orth

def get_savepath(id, i, save_folders, n_arms, suffix):
    filename = f'test-save-{id}-{i:03d}-{suffix}.png'
    save_folder = save_folders[suffix]
    folder = os.path.join(save_folder, str(n_arms))
    os.makedirs(folder, exist_ok=True)
    savepath = os.path.join(folder, filename)
    return savepath

def create_subfolders(root, subfolder_names):
    save_folders = {} # dictionary that holds subfolder paths
    for f in subfolder_names:
        sub_path = os.path.join(root, f)
        os.makedirs(sub_path, exist_ok=True)
        save_folders[f] = sub_path
    return save_folders
    
def process_stl(file_path, save_dir, n_proj, params):
    # plotter/screenshot parameters
    res = 224
    bg_color='black'
    obj_color='white'
    op=1.0
    theta_2ds, theta_phips = 90, 120
    mesh = pv.read(file_path) # read stl file as mesh
    id = file_path.rsplit('-',1)[1].rsplit('.',1)[0] # get id as string
    n_arms = params[int(id)][0][5]
    # create subfolders for saving
    subfolders = ['default', '2ds', 'phips']
    save_folders = create_subfolders(save_dir, subfolders)
    start_time = time.time() # for testing
    render_time = 0
    save_time = 0
    remove_time = 0
    for i in range(n_proj):
        pl = init_plotter(res, bg_color) # initiate plotter
        # === DEFAULT === #
        rotated_mesh = random_rotate(mesh)
        actor_default = pl.add_mesh(rotated_mesh, show_edges=None, 
                                    color=obj_color, opacity=op)
        camera_position_default = pl.camera.position
        axis_rotation = get_orthogonal_vector(camera_position_default)
        savepath = get_savepath(id, i, save_folders, n_arms, 'default')
        pl.render()
        start_save = time.time()
        pl.screenshot(savepath, return_img=False)
        end_save = time.time()
        save_time += end_save - start_save
        pl.remove_actor(actor_default)
        pl.close()
    end_time = time.time() # for testing
    print(f'processing {file_path}: {end_time - start_time} seconds')
    print(f'-> time to save: {save_time} seconds')

def main():
    print("this is a test print: inside of python script")
    # user defined parameters 
    num_cores =  int(sys.argv[1]) # these based on PBS script
    n_proj = int(sys.argv[2])  
    save_dir = sys.argv[3]
    os.makedirs(save_dir, exist_ok=True)
    # chunk = int(sys.argv[4])
    # Load the JSON file
    params_path = '/glade/u/home/joko/ice3d/output/params_200_50.json'
    with open(params_path, 'rb') as file:
        params = json.load(file)
    # load list of STL filepaths
    stl_paths_txt = '/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/stl/stl_relative_paths.txt'
    with open(stl_paths_txt, 'r') as file:
        basepath = '/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/stl'
        rel_paths = [line.strip().replace('./','') for line in file]
        stl_paths = [os.path.join(basepath, i) for i in rel_paths]
    stl_paths=stl_paths[:100]
    # # chunk stl_paths
    # print(f'stl_paths length: {len(stl_paths)}')
    # start_idx = int((chunk-1)*(len(stl_paths)/7))
    # end_idx = int(chunk*(len(stl_paths)/7))
    # print(f'Processing index range: [{start_idx}:{end_idx}]')
    # print(f'index dtype: {type(start_idx)}, {type(end_idx)}')
    # stl_paths = stl_paths[start_idx:end_idx]
    # sys.stdout.flush() # force write to stdout
    for file_path in stl_paths:
        process_stl(file_path, save_dir, n_proj, params)

if __name__ == "__main__":
    main()