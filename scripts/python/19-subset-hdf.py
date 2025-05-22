import h5py
import argparse
import sys

def subset_hdf(input_path, output_path, subset_size, chunk_size):
    with h5py.File(input_path, 'r') as infile, h5py.File(output_path, 'w') as outfile:
        for dset_name in ['images', 'n_arms', 'rho_eff', 'sa_eff', 'filenames']:
            data = infile[dset_name][:subset_size]
            # Set chunk shape: for 1D, it's (chunk_size,), for ND, chunk along first axis
            shape = data.shape
            if len(shape) == 1:
                chunks = (min(chunk_size, shape[0]),)
            else:
                chunks = (min(chunk_size, shape[0]),) + shape[1:]
            outfile.create_dataset(dset_name, data=data, chunks=chunks)

def qa_check(input_path, output_path, subset_size):
    with h5py.File(input_path, 'r') as infile, h5py.File(output_path, 'r') as outfile:
        for dset_name in ['images', 'n_arms', 'rho_eff', 'sa_eff', 'filenames']:
            in_shape = infile[dset_name].shape[0]
            out_shape = outfile[dset_name].shape[0]
            expected = min(subset_size, in_shape)
            if out_shape != expected:
                print(f"QA check failed for {dset_name}: expected {expected}, got {out_shape}")
                return False
    print("QA check passed: All datasets have the correct subset size.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset an HDF5 file with chunking.")
    parser.add_argument("--input_hdf", help="Path to input HDF5 file")
    parser.add_argument("--output_hdf", help="Path to output HDF5 file")
    parser.add_argument("--subset_size", type=int, help="Number of elements to include in the subset")
    parser.add_argument("--chunk_size", type=int, default=100, help="Chunk size along the first dimension (default: 100)")
    args = parser.parse_args()
    subset_hdf(args.input_hdf, args.output_hdf, args.subset_size, args.chunk_size)
    if not qa_check(args.input_hdf, args.output_hdf, args.subset_size):
        sys.exit(1)