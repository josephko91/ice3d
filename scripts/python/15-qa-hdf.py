import time
import pandas as pd
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def verify_hdf5_file(hdf5_path, df, sample_count=5):
    print(f"🔍 Verifying HDF5 file: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Check datasets
        assert 'images' in f and 'filenames' in f and 'n_arms' in f and 'rho_eff' in f and 'sa_eff' in f, "Missing required datasets"
        images = f['images']
        filenames = f['filenames']
        n_arms = f['n_arms']
        rho_eff = f['rho_eff']
        sa_eff = f['sa_eff']

        print(f"✅ Datasets found. Shape: images={images.shape}, labels={filenames.shape}, n_arms={n_arms.shape}, rho_eff={rho_eff.shape}, sa_eff={sa_eff.shape}")
        print(f"   Image dtype: {images.dtype}, Filenames dtype: {filenames.dtype}")
        print(f"   n_arms dtype: {n_arms.dtype}, rho_eff dtype: {rho_eff.dtype}, sa_eff dtype: {sa_eff.dtype}")
        print(f"   Image shape: {images.shape}, Filenames shape: {filenames.shape}")
        print(f"   n_arms shape: {n_arms.shape}, rho_eff shape: {rho_eff.shape}, sa_eff shape: {sa_eff.shape}")

        assert len(images) == len(filenames), "Mismatch in number of images and labels"

        # Sample a few entries
        for i in range(min(sample_count, len(images))):
            img = images[i]
            label = filenames[i].decode('utf-8') if isinstance(filenames[i], bytes) else filenames[i]
            n_arms_sample = n_arms[i]
            rho_eff_sample = rho_eff[i]
            sa_eff_sample = sa_eff[i]

            print(f"Sample {i}: label='{label}', image shape={img.shape}, dtype={img.dtype}")
            
            # Get ground truth values from the dataframe
            ground_truth = df[df['filename'] == label]
            if not ground_truth.empty:
                gt_n_arm = ground_truth['n_arms'].values[0]
                gt_rho = ground_truth['rho_eff'].values[0]
                gt_sa = ground_truth['sa_eff'].values[0]
                print(f"   Ground truth: n_arms={gt_n_arm}, rho_eff={gt_rho}, sa_eff={gt_sa}")
                print(f"   HDF5 values: n_arms={n_arms_sample}, rho_eff={rho_eff_sample}, sa_eff={sa_eff_sample}")
                # Compare values
                assert np.isclose(n_arms_sample, gt_n_arm, atol=1e-6), f"Mismatch in n_arms for {label}: HDF5={n_arms_sample}, Ground Truth={gt_n_arm}"
                assert np.isclose(rho_eff_sample, gt_rho, atol=1e-6), f"Mismatch in rho_eff for {label}: HDF5={rho_eff_sample}, Ground Truth={gt_rho}"
                assert np.isclose(sa_eff_sample, gt_sa, atol=1e-6), f"Mismatch in sa_eff for {label}: HDF5={sa_eff_sample}, Ground Truth={gt_sa}"
                print(f"   Ground truth match for {label}: n_arms={gt_n_arm}, rho_eff={gt_rho}, sa_eff={gt_sa}")
            else:
                print(f"   Warning: No ground truth found for {label}")

    print("✅ QA check completed successfully.\n")

# Example usage
hdf5_file = "/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/imgs-ml-ready/2ds.h5"
data_dir = '/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/tabular-data-v2/ros-tabular-data.parquet'
start_time = time.time()
df = pd.read_parquet(data_dir)
elapsed_time = time.time() - start_time
print(f"Time taken to read parquet file: {elapsed_time:.2f} seconds")
sample_count = 10
verify_hdf5_file(hdf5_file, df, sample_count=sample_count)
