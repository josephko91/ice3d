import pandas as pd
import numpy as np
import h5py
import time

# set directories
data_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/tabular-data-v2/ros-tabular-data.parquet'
hdf5_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/2ds.h5'

#read files
start_time = time.time()
df = pd.read_parquet(data_dir)
elapsed_time = time.time() - start_time
print(f"Time taken to read parquet file: {elapsed_time:.2f} seconds")

start_time = time.time()
with h5py.File(hdf5_dir, 'r') as hdf:
    print("Reading all filenames...")
    try:
        file_id = hdf['filenames'][:]  # Read all filenames
        print(f"All filenames read successfully. First 10 entries: {file_id[:10]}")
    except Exception as e:
        print(f"Error reading filenames: {e}")

elapsed_time = time.time() - start_time
print(f"Time taken to read all filenames: {elapsed_time:.2f} seconds")

# Convert file_id (bytes) to string if needed
file_id_str = np.array([fid.decode() if isinstance(fid, bytes) else fid for fid in file_id])
print('checkpoint 1')

# # Remove '.png' from df['filename'] to match file_id format
# df['file_id_base'] = df['filename'].str.replace('.png', '', regex=False)
# print('checkpoint 2')

# Create a DataFrame indexed by file_id_base for fast lookup
df_indexed = df.set_index('filename')
print('checkpoint 2')

# get labels (rho_eff, sa_eff, n_arms)
print('getting labels....')
rho_eff = df_indexed.loc[file_id_str, 'rho_eff'].values
sa_eff = df_indexed.loc[file_id_str, 'sa_eff'].values
n_arms = df_indexed.loc[file_id_str, 'n_arms'].values
print('finished getting labels!')

# print size of rho_eff, sa_eff, n_arms
print(f"rho_eff size: {rho_eff.size}")
print(f"sa_eff size: {sa_eff.size}")
print(f"n_arms size: {n_arms.size}")

# add rho_eff, sa_eff, and n_arms as separate datasets in the hdf5 file
with h5py.File(hdf5_dir, 'a') as hdf:
    # Create datasets only if they don't already exist
    for name, data in zip(['rho_eff', 'sa_eff', 'n_arms'], [rho_eff, sa_eff, n_arms]):
        if name in hdf:
            print(f"Dataset {name} already exists. Deleting it...")
            del hdf[name]  # Delete existing dataset
        print(f"Creating dataset: {name}")
        start_time = time.time()
        hdf.create_dataset(name, data=data, dtype='f8')
        elapsed_time = time.time() - start_time
        print(f"Dataset {name} created in {elapsed_time:.2f} seconds")

# QA check to ensure the lengths match
for name, data in zip(['rho_eff', 'sa_eff', 'n_arms'], [rho_eff, sa_eff, n_arms]):
    print(f"Checking length of {name}...")
    if len(data) != len(file_id):
        raise ValueError(f"Length mismatch for {name}: expected {len(file_id)}, got {len(data)}")
    print(f"Length check passed for {name}!")