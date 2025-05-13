import os
import h5py
import numpy as np
from PIL import Image
import glob
import time
from multiprocessing import Pool

def process_image(args):
    """
    Loads a single image and converts it to grayscale uint8.
    Args:
        args: tuple of (index, image_file)
    Returns:
        tuple: (index, image_array, label_bytes)
    """
    index, image_file = args
    try:
        img = Image.open(image_file).convert('L')            # single-channel
        img_array = np.array(img, dtype=np.uint8)
        label = os.path.splitext(os.path.basename(image_file))[0].encode('utf-8')
        return (index, img_array, label)
    except Exception as e:
        print(f"Failed to process {image_file}: {e}")
        return None

def create_hdf5_streaming_multiprocessing(image_folder,
                                          output_file,
                                          batch_size=32,
                                          progress_interval=1000,
                                          num_workers=8):
    """
    Uses multiprocessing to read images and writes them in batches to an HDF5 file with chunking.
    """
    start_time = time.time()

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

    # Create the HDF5 file with batch-based chunking
    with h5py.File(output_file, 'w') as f:
        images_dset = f.create_dataset(
            'images',
            shape=(total_images, *img_shape),
            dtype='uint8',
            chunks=(batch_size, *img_shape)
        )
        labels_dset = f.create_dataset(
            'labels',
            shape=(total_images,),
            dtype='S256',
            chunks=(batch_size,)
        )

        print(f"Starting multiprocessing pool with {num_workers} workers...")
        image_args = [(idx, path) for idx, path in enumerate(all_image_files)]

        buffer = []
        processed = 0

        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(process_image, image_args, chunksize= batch_size):
                if result is None:
                    continue

                buffer.append(result)
                processed += 1

                # When we have a full batch, write it out
                if len(buffer) >= batch_size:
                    indices, imgs, labels = zip(*buffer)
                    images_dset[list(indices), ...] = np.stack(imgs)
                    labels_dset[list(indices)] = labels
                    buffer.clear()

                # Progress report
                if processed % progress_interval == 0:
                    print(f"Processed {processed}/{total_images} images...")

        # Flush any leftovers in buffer
        if buffer:
            indices, imgs, labels = zip(*buffer)
            images_dset[list(indices), ...] = np.stack(imgs)
            labels_dset[list(indices)] = labels
            buffer.clear()

    elapsed = time.time() - start_time
    print(f"\n✅ Done. Saved HDF5 to: {output_file}")
    print(f"⏱ Total time: {elapsed:.2f} seconds")


# ---- CONFIGURATION ----
image_folder     = "/glade/derecho/scratch/joko/synth-ros/params_200_50_20250403/projections/default"
output_file      = "/glade/derecho/scratch/joko/params_200_50_20250403_default.h5"
batch_size       = 64
progress_interval= 1000
num_workers      = 62   # match your available CPU cores

# ---- RUN ----
create_hdf5_streaming_multiprocessing(
    image_folder,
    output_file,
    batch_size=batch_size,
    progress_interval=progress_interval,
    num_workers=num_workers
)