import h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Reorder HDF5 file based on external filename order.")
    parser.add_argument('--hdf_file', type=str, required=True, help='Input HDF5 file path')
    parser.add_argument('--filename_order', type=str, required=True, help='Text file with desired filename order (one per line)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output HDF5 file')
    parser.add_argument('--num_cpus', type=int, default=8, help='Number of parallel workers')
    return parser.parse_args()

def process_chunk(args):
    chunk_indices, chunk_out_start, hdf_file_path = args
    print(f"[process_chunk] Reading chunk: start={chunk_out_start}, len={len(chunk_indices)}", flush=True)
    with h5py.File(hdf_file_path, 'r') as f:
        images = f['images'][chunk_indices]
        n_arms = f['n_arms'][chunk_indices]
        rho_eff = f['rho_eff'][chunk_indices]
        sa_eff = f['sa_eff'][chunk_indices]
        filenames = f['filenames'][chunk_indices]
    print(f"[process_chunk] Finished chunk: start={chunk_out_start}", flush=True)
    return (chunk_out_start, images, n_arms, rho_eff, sa_eff, filenames)

def main():
    args = parse_args()

    print(f"[main] Loading filename order from: {args.filename_order}")
    with open(args.filename_order, 'r') as f:
        filenames_default = [line.strip() for line in f if line.strip()]
    filenames_default = np.array(filenames_default)
    print(f"[main] Number of filenames in order file: {len(filenames_default)}")

    print(f"[main] Reading filenames from HDF5: {args.hdf_file}")
    with h5py.File(args.hdf_file, 'r') as f:
        filenames_h5 = f['filenames'][:].astype(str)
    print(f"[main] Number of filenames in HDF5: {len(filenames_h5)}")

    print("[main] Building filename to index mapping")
    filename_to_index = {name: i for i, name in enumerate(filenames_h5)}
    print(f"[main] Example mapping: {list(filename_to_index.items())[:3]}")

    print("[main] Mapping external order to HDF5 indices")
    indices = np.array([filename_to_index[name] for name in filenames_default])
    print(f"[main] First 5 mapped indices: {indices[:5]}")

    print("[main] Sorting indices for h5py fancy indexing")
    sort_idx = np.argsort(indices)
    indices_sorted = indices[sort_idx]
    print(f"[main] First 5 sorted indices: {indices_sorted[:5]}")

    num_samples = len(indices_sorted)
    num_workers = args.num_cpus
    chunk_size = num_samples // num_workers
    print(f"[main] Preparing {num_workers} chunks, chunk_size={chunk_size}")
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_workers - 1 else num_samples
        print(f"[main] Chunk {i}: start={start}, end={end}")
        chunks.append((indices_sorted[start:end], start, args.hdf_file))

    print("[main] Starting parallel read")
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)
    print("[main] Parallel read complete")

    # Concatenate results in sorted order
    images_sorted = []
    n_arms_sorted = []
    rho_eff_sorted = []
    sa_eff_sorted = []
    filenames_sorted = []
    for _, images, n_arms, rho_eff, sa_eff, filenames in sorted(results, key=lambda x: x[0]):
        images_sorted.append(images)
        n_arms_sorted.append(n_arms)
        rho_eff_sorted.append(rho_eff)
        sa_eff_sorted.append(sa_eff)
        filenames_sorted.append(filenames)
    images_sorted = np.concatenate(images_sorted, axis=0)
    n_arms_sorted = np.concatenate(n_arms_sorted, axis=0)
    rho_eff_sorted = np.concatenate(rho_eff_sorted, axis=0)
    sa_eff_sorted = np.concatenate(sa_eff_sorted, axis=0)
    filenames_sorted = np.concatenate(filenames_sorted, axis=0)

    # Now, unsort to match the original requested order
    unsort_idx = np.argsort(sort_idx)
    images_final = images_sorted[unsort_idx]
    n_arms_final = n_arms_sorted[unsort_idx]
    rho_eff_final = rho_eff_sorted[unsort_idx]
    sa_eff_final = sa_eff_sorted[unsort_idx]
    filenames_final = filenames_sorted[unsort_idx]

    os.makedirs(args.save_dir, exist_ok=True) # ensure dir exists
    base_name = os.path.splitext(os.path.basename(args.hdf_file))[0]
    output_suffix = os.path.basename(os.path.normpath(args.save_dir))
    output_h5 = os.path.join(args.save_dir, f"{base_name}_{output_suffix}.h5")
    print(f"[main] Writing output to: {output_h5}")
    # Write to output in the correct order
    with h5py.File(output_h5, 'w') as f_out:
        with h5py.File(args.hdf_file, 'r') as f_in:
            img_shape = f_in['images'].shape[1:]
        f_out.create_dataset('images', data=images_final, dtype='uint8')
        f_out.create_dataset('n_arms', data=n_arms_final, dtype='f8')
        f_out.create_dataset('rho_eff', data=rho_eff_final, dtype='f8')
        f_out.create_dataset('sa_eff', data=sa_eff_final, dtype='f8')
        f_out.create_dataset('filenames', data=filenames_final, dtype='S256')
    print("[main] Done.")

if __name__ == "__main__":
    main()