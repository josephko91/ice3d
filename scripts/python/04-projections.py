import pyvista as pv
import os, random
import numpy as np
import miniball

def random_rotate(mesh):
    """
    Rotate rosette in a random orientation
    TODO:
    - fix bug related to the reliance on model attribute
    """
    rotated = mesh.copy()
    deg_x = np.random.randint(1, 360)
    deg_y = np.random.randint(1, 360)
    deg_z = np.random.randint(1, 360)
    rotated_model = rotated.rotate_x(deg_x, inplace=False)
    rotated_model.rotate_y(deg_y, inplace=True)
    rotated_model.rotate_z(deg_z, inplace=True)
    return rotated_model

def projection():
    pass
    
def process_chunk():
    pass
    
def main():
    # set directories

    # load list of 

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

    

if __name__ == "__main__":
    main()