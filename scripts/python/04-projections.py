import pyvista as pv
import os, sys
import random
import numpy as np
import json

def random_rotate(mesh):
    rotated = mesh.copy()
    deg_x = np.random.randint(1, 360)
    deg_y = np.random.randint(1, 360)
    deg_z = np.random.randint(1, 360)
    rotated.rotate_x(deg_x, inplace=True)
    rotated.rotate_y(deg_y, inplace=True)
    rotated.rotate_z(deg_z, inplace=True)
    return rotated

def process_instance(stl_path, save_dir, n_proj, params):
    res = 128
    bg_color='black'
    obj_color='white'
    op=1.0
    pl = pv.Plotter(off_screen=True, window_size=[res, res])
    pl.background_color = bg_color
    pl.enable_parallel_projection()
    pl.remove_all_lights()
    mesh = pv.read(stl_path) # read stl file as mesh
    id = stl_path.rsplit('-',1)[1].rsplit('.',1)[0] # get id as string
    n_arms = params[int(id)][0][5]
    save_folder = os.path.join(save_dir, str(n_arms))
    os.makedirs(save_folder, exist_ok=True)
    for i in range(n_proj):
        rotated_mesh = random_rotate(mesh)
        pl.add_mesh(rotated_mesh, show_edges=None, color = obj_color, opacity=op, name='mesh')
        file_name = f'ros-projection-{id}-{i:03d}.png'
        file_path = os.path.join(save_folder, file_name)
        pl.screenshot(file_path, return_img=False)
    pl.close()
    
def process_chunk(chunk, start_index, end_index, save_dir, n_proj, params):
    for i in range(start_index, end_index):
        stl_path = chunk[i]
        process_instance(stl_path, save_dir, n_proj, params)
        
def main():
    # set number of random projections
    n_proj = 100
    # set directories
    save_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/projections'
    # Load the JSON file
    params_path = '/glade/u/home/joko/ice3d/output/params_200_50.json'
    with open(params_path, 'rb') as file:
        params = json.load(file)
    # load list of STL filepaths
    stl_paths_txt = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/stl/stl_relative_paths.txt'
    with open(stl_paths_txt, 'r') as file:
        basepath = '/glade/derecho/scratch/joko/synth-ros/params_200_50-debug-20250316/stl'
        rel_paths = [line.strip().replace('./','') for line in file]
        stl_paths = [os.path.join(basepath, i) for i in rel_paths]
    # Get the total number of tasks (this will be passed by PBS)
    num_tasks = int(sys.argv[1])
    # Get the job index from PBS (PBS_ARRAYID is the index for each job in the array)
    task_index = int(sys.argv[2])
    # Calculate the chunk size for each task
    chunk_size = len(stl_paths) // num_tasks  # Integer division
    remainder = len(stl_paths) % num_tasks
    # Calculate the start and end indices for the current task
    start_index = task_index * chunk_size
    if (task_index == (num_tasks-1)) & (remainder > 0):
        end_index = start_index + chunk_size + remainder 
    else:
        end_index = start_index + chunk_size
    # check if in index, process data chunk
    if start_index < len(stl_paths):
        print(f'processing chunk {start_index}:{end_index}')
        process_chunk(stl_paths, start_index, end_index, save_dir, n_proj, params)
    else:
        sys.exit() # out of index

if __name__ == "__main__":
    main()