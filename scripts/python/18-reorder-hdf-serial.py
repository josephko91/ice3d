import h5py
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Reorder HDF5 file based on external filename order.")
    parser.add_argument('--hdf_file', type=str, required=True, help='Input HDF5 file path')
    parser.add_argument('--filename_order', type=str, required=True, help='Text file with desired filename order (one per line)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output HDF5 file')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Chunk size for writing')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"[main] Loading filename order from: {args.filename_order}")
    with open(args.filename_order, 'r') as f:
        filenames_default = [line.strip() for line in f if line.strip()]
    filenames_default = np.array(filenames_default)
    print(f"[main] Number of filenames in order file: {len(filenames_default)}")

    print(f"[main] Reading filenames from HDF5: {args.hdf_file}")
    with h5py.File(args.hdf_file, 'r') as f:
        num_samples = f['filenames'].shape[0]
        chunk_size = args.chunk_size
        filenames_h5 = []
        images = []
        n_arms = []
        rho_eff = []
        sa_eff = []
        for i in range(0, num_samples, chunk_size):
            sl = slice(i, min(i + chunk_size, num_samples))
            filenames_h5.append(f['filenames'][sl].astype(str))
            images.append(f['images'][sl])
            n_arms.append(f['n_arms'][sl])
            rho_eff.append(f['rho_eff'][sl])
            sa_eff.append(f['sa_eff'][sl])
        filenames_h5 = np.concatenate(filenames_h5, axis=0)
        images = np.concatenate(images, axis=0)
        n_arms = np.concatenate(n_arms, axis=0)
        rho_eff = np.concatenate(rho_eff, axis=0)
        sa_eff = np.concatenate(sa_eff, axis=0)
    print(f"[main] Number of filenames in HDF5: {len(filenames_h5)}")

    print("[main] Building filename to index mapping")
    filename_to_index = {name: i for i, name in enumerate(filenames_h5)}
    print(f"[main] Example mapping: {list(filename_to_index.items())[:3]}")

    print("[main] Mapping external order to HDF5 indices")
    missing = [name for name in filenames_default if name not in filename_to_index]
    if missing:
        raise ValueError(f"Missing filenames in HDF5: {missing[:5]} ... (total {len(missing)})")
    indices = np.array([filename_to_index[name] for name in filenames_default])
    print(f"[main] First 5 mapped indices: {indices[:5]}")

    print("[main] Reordering datasets to match requested order")
    images_final = images[indices]
    n_arms_final = n_arms[indices]
    rho_eff_final = rho_eff[indices]
    sa_eff_final = sa_eff[indices]
    filenames_final = filenames_h5[indices].astype('S256')
    print('Done reordering datasets!')

    os.makedirs(args.save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.hdf_file))[0]
    output_suffix = os.path.basename(os.path.normpath(args.save_dir))
    output_h5 = os.path.join(args.save_dir, f"{base_name}_{output_suffix}.h5")
    print(f"[main] Writing output to: {output_h5}")
    with h5py.File(output_h5, 'w') as f_out:
        img_shape = images.shape[1:]
        chunk_size = args.chunk_size
        print('Writing images with shape:', img_shape, 'and chunk size:', chunk_size)
        # Create datasets with chunking
        f_out.create_dataset(
            'images',
            data=images_final,
            dtype='uint8',
            chunks=(chunk_size, *img_shape)
        )
        print('Writing n_arms, rho_eff, sa_eff, filenames datasets')
        f_out.create_dataset(
            'n_arms',
            data=n_arms_final,
            dtype='f8',
            chunks=(chunk_size,)
        )
        f_out.create_dataset(
            'rho_eff',
            data=rho_eff_final,
            dtype='f8',
            chunks=(chunk_size,)
        )
        f_out.create_dataset(
            'sa_eff',
            data=sa_eff_final,
            dtype='f8',         
            chunks=(chunk_size,)
        )
        f_out.create_dataset(
            'filenames',
            data=filenames_final,
            dtype='S256',
            chunks=(chunk_size,)
        )
    print("[main] Done.")

if __name__ == "__main__":
    main()