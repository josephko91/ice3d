import os
import h5py
import numpy as np
from PIL import Image
import glob
import time
from multiprocessing import Pool
import argparse

def process_image(args):
    """
    Loads a single image and converts it to grayscale uint8.
    Args:
        args: tuple of (index, image_file)
    Returns:
        tuple: (index, image_array, filename_bytes)
    """
    index, image_file = args
    try:
        img = Image.open(image_file).convert('L')            # single-channel
        img_array = np.array(img, dtype=np.uint8)
        filename = os.path.basename(image_file).encode('utf-8')  # include extension
        return (index, img_array, filename)
    except Exception as e:
        print(f"Failed to process {image_file}: {e}")
        return None

def create_hdf5_per_core(image_args, output_file, img_shape, batch_size):
    """
    Optimized: Creates an HDF5 file for a subset of images processed by a single core.
    Improvements:
        - Avoids redundant processing
        - Reads all data first, then writes in bulk
        - Enables compression (optional, but good for large files)
    """
    print(f"🧵 Starting: {output_file}")
    start_time = time.time()

    processed_imgs = []
    processed_filenames = []

    for i, args in enumerate(image_args):
        result = process_image(args)
        if result is None:
            continue
        _, img_array, filename = result
        processed_imgs.append(img_array)
        processed_filenames.append(filename)

        # Print progress every N iterations
        if i % 1_000 == 0:  # Adjust N as needed (e.g., 100)
            print(f"Processed {i}/{len(image_args)} images...")

    if len(processed_imgs) == 0:
        print(f"⚠️ No valid images for: {output_file}")
        return

    processed_imgs = np.stack(processed_imgs)  # shape: (N, H, W)
    processed_filenames = np.array(processed_filenames, dtype='S256')  # shape: (N,)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset(
            'images',
            data=processed_imgs,
            dtype='uint8',
            chunks=(batch_size, *img_shape)
        )
        f.create_dataset(
            'filenames',
            data=processed_filenames,
            dtype='S256',
            chunks=(batch_size,)
        )

    elapsed = time.time() - start_time
    print(f"✅ Finished: {output_file} ({len(processed_imgs)} samples) in {elapsed:.2f}s")


def merge_hdf5_files(temp_files, final_output_file, img_shape, batch_size):
    """
    Merges multiple HDF5 files into a single final HDF5 file.
    """
    print("Calculating total number of images across temporary files...")
    total_images = sum(h5py.File(f, 'r')['images'].shape[0] for f in temp_files)
    print(f"Total images to merge: {total_images}")

    with h5py.File(final_output_file, 'w') as f:
        print("Creating datasets in the final HDF5 file...")
        images_dset = f.create_dataset(
            'images',
            shape=(total_images, *img_shape),
            dtype='uint8',
            chunks=(batch_size, *img_shape)
        )
        filenames_dset = f.create_dataset(
            'filenames',
            shape=(total_images,),
            dtype='S256',
            chunks=(batch_size,)
        )

        start_idx = 0
        for i, temp_file in enumerate(temp_files):
            print(f"Processing temporary file {i + 1}/{len(temp_files)}: {temp_file}")
            with h5py.File(temp_file, 'r') as temp_f:
                temp_images = temp_f['images'][:]
                temp_filenames = temp_f['filenames'][:]

                end_idx = start_idx + temp_images.shape[0]
                print(f"Writing images {start_idx} to {end_idx - 1} into the final HDF5 file...")
                images_dset[start_idx:end_idx, ...] = temp_images
                filenames_dset[start_idx:end_idx] = temp_filenames
                start_idx = end_idx

    print("Merging complete. Final HDF5 file created.")

def create_hdf5_streaming_multiprocessing(image_folder,
                                          final_output_file,
                                          batch_size=32,
                                          progress_interval=1000,
                                          num_workers=8,
                                          temp_dir="."):
    """
    Uses multiprocessing to read images, creates HDF5 files per core, and merges them into a final HDF5 file.
    """
    start_time = time.time()

    # Ensure temp_dir exists
    os.makedirs(temp_dir, exist_ok=True)

    # Gather all image files
    all_image_files = []
    for subdir in sorted(os.listdir(image_folder)):
        subdir_path = os.path.join(image_folder, subdir)
        if os.path.isdir(subdir_path):
            all_image_files.extend(glob.glob(os.path.join(subdir_path, "*.png")))

    total_images = len(all_image_files)
    if total_images == 0:
        print("No images found.")
        return

    print(f"Found {total_images} images. Reading sample image for dimensions...")

    # Determine image shape
    sample_image = Image.open(all_image_files[0]).convert('L')
    img_shape = np.array(sample_image).shape  # (height, width)

    # Split image files into chunks for each worker
    image_args = [(idx, path) for idx, path in enumerate(all_image_files)]
    chunks = np.array_split(image_args, num_workers)

    print(f"Starting multiprocessing pool with {num_workers} workers...")
    temp_files = []

    with Pool(processes=num_workers) as pool:
        temp_files = [os.path.join(temp_dir, f"temp_worker_{i}.h5") for i in range(len(chunks))]
        pool.starmap(create_hdf5_per_core, [(chunk, temp_file, img_shape, batch_size) for chunk, temp_file in zip(chunks, temp_files)])

    print("Merging temporary HDF5 files into final output...")
    merge_hdf5_files(temp_files, final_output_file, img_shape, batch_size)

    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

    elapsed = time.time() - start_time
    print(f"\n✅ Done. Saved final HDF5 to: {final_output_file}")
    print(f"⏱ Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HDF5 file from images.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("final_output_file", type=str, help="Path to the final output HDF5 file.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images.")
    parser.add_argument("--progress_interval", type=int, default=1000, help="Progress reporting interval.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--temp_dir", type=str, default=".", help="Directory to store temporary files.")

    args = parser.parse_args()

    create_hdf5_streaming_multiprocessing(
        args.image_folder,
        args.final_output_file,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
        num_workers=args.num_workers,
        temp_dir=args.temp_dir
    )