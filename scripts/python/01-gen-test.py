import psutil
import numpy as np
from data.rosette_temp import Rosette
import pandas as pd
from multiprocess import Pool, get_context
from itertools import product
import os

# Main driver of script
def main():
    a = 1.5
    c = 3.5
    r0 = 1.0
    h0 = 0.25
    hp = 0.75
    n_arms = 6
    ros = Rosette(a, c, r0, h0, hp, n_arms)
    ros.unify_mesh()
    ros_file_name = 'test-rosette-02.stl'
    save_path = '/home/jko/synthetic/models'
    ros_file_path = os.path.join(save_path, ros_file_name)
    ros.model.save(ros_file_path)
# protect entry point and run main
if __name__ == "__main__":
    main()