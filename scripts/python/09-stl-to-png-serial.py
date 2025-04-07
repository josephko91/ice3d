import pyvista as pv
import os, sys
import random
import numpy as np
import json
import time
'''
This script does the following:
- read mesh (STL) and take three different projections
- projection 1: default view
- projection 2: 2D-S view (90 degrees from default view)
- projection 3: PHIPS view (120 degrees from default view)
- saves each projection as png file 
- Note: this version is based on rotation of mesh, not moving the camera
'''
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
    return v_orthimport os
import multiprocessing
import pyvista as pv

# Example function to process a single STL file
def process_stl(file_path):
    # Read the STL file using PyVista
    mesh = pv.read(file_path)
    
    # Perform operations on the mesh (for example, compute the volume)
    # You can replace this with any other operations that you need
    volume = mesh.volume
    print(f"Processed {file_path} with volume {volume}")
    
    # You could save the results or perform other operations here
    # For example, return the result or save the processed mesh to a file
    # mesh.save(f"processed_{os.path.basename(file_path)}")
    return volume

def main():
    # Example list of STL file paths (replace with your actual list)
    stl_paths = ["path/to/file1.stl", "path/to/file2.stl", "path/to/file3.stl"]
    
    # Get the number of CPU cores available on the system
    num_cores = multiprocessing.cpu_count()

    # Use a multiprocessing Pool to process the STL files in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Map the process_stl function to the list of file paths
        results = pool.map(process_stl, stl_paths)

    # Print or save the results (you can modify this as needed)
    print(f"Processed results: {results}")

if __name__ == "__main__":
    main()


def random_rotate(mesh):
    rotated = mesh.copy()
    deg_x = np.random.randint(1, 360)
    deg_y = np.random.randint(1, 360)
    deg_z = np.random.randint(1, 360)
    rotated.rotate_x(deg_x, inplace=True)
    rotated.rotate_y(deg_y, inplace=True)
    rotated.rotate_z(deg_z, inplace=True)
    return rotated

def init_plotter(res, bg_color):
    pl = pv.Plotter(off_screen=True, window_size=[res, res])
    pl.background_color = bg_color
    pl.enable_parallel_projection()
    pl.remove_all_lights()
    return pl 

def get_savepath(id, i, save_folders, n_arms, suffix):
    filename = f'ros-projection-{id}-{i:03d}-{suffix}.png'
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

def process_instance(stl_path, save_dir, n_proj, params):
    # projection parameters 
    res = 224
    bg_color='black'
    obj_color='white'
    op=1.0
    theta_2ds, theta_phips = 90, 120
    mesh = pv.read(stl_path) # read stl file as mesh
    id = stl_path.rsplit('-',1)[1].rsplit('.',1)[0] # get id as string
    n_arms = params[int(id)][0][5]
    # create subfolders for saving
    subfolders = ['default', '2ds', 'phips']
    save_folders = create_subfolders(save_dir, subfolders)
    # subfolder = os.path.join(save_dir, str(task_index))
    # os.makedirs(subfolder, exist_ok=True)
    # save_folder = os.path.join(save_dir, str(n_arms))
    # os.makedirs(save_folder, exist_ok=True)
    start_time = time.time() # for testing
    render_time = 0
    save_time = 0
    remove_time = 0
    for i in range(n_proj):
        pl = init_plotter(res, bg_color) # initiate plotter
        ### default projection
        rotated_mesh = random_rotate(mesh)
        actor_default = pl.add_mesh(rotated_mesh, show_edges=None, color = obj_color, opacity=op)
        camera_position_default = pl.camera.position
        axis_rotation = get_orthogonal_vector(camera_position_default) # get axis of rotation
        savepath = get_savepath(id, i, save_folders, n_arms, 'default')
        start_render = time.time()
        pl.render()
        end_render = time.time()
        render_time += end_render - start_render
        start_save = time.time()
        pl.screenshot(savepath, return_img=False)
        end_save = time.time()
        save_time += end_save - start_save
        start_remove = time.time()
        pl.remove_actor(actor_default) # remove default
        end_remove = time.time()
        remove_time += end_remove - start_remove
        ### 2D-S (90deg), first view
        mesh_2ds = rotated_mesh.rotate_vector(axis_rotation, theta_2ds, point=rotated_mesh.center)
        actor_2ds = pl.add_mesh(mesh_2ds, show_edges=None, color = obj_color, opacity=op)
        savepath = get_savepath(id, i, save_folders, n_arms, '2ds')
        pl.render()
        pl.screenshot(savepath, return_img=False)
        pl.remove_actor(actor_2ds) # remove 2ds 
        ### PHIPS-HALO (120deg), second view
        mesh_phips = rotated_mesh.rotate_vector(axis_rotation, theta_phips, point=rotated_mesh.center)
        pl.add_mesh(mesh_phips, show_edges=None, color = obj_color, opacity=op)
        savepath = get_savepath(id, i, save_folders, n_arms, 'phips')
        pl.render()
        pl.screenshot(savepath, return_img=False)
        pl.close()
    end_time = time.time() # for testing
    print(f'processing {stl_path}: {end_time - start_time} seconds')
    print(f'-> time to render: {render_time} seconds')
    print(f'-> time to save: {save_time} seconds')
    print(f'-> time to remove actors: {remove_time} seconds')
        
def main():
    # set number of random projections
    n_proj = 100
    n_random = 10
    # set directories
    save_dir = '/glade/derecho/scratch/joko/synth-ros/test-stereo-projections/serial-n10'
    os.makedirs(save_dir, exist_ok=True)
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
    stl_paths = stl_paths[:n_random] # FOR TESTING (get small subset)
    # process stl files iteratively 
    for f in stl_paths:
        process_instance(f, save_dir, n_proj, params)

if __name__ == "__main__":
    main()