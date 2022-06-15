from typing import List
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from collections import Counter
from utils.pyutils import crop_validation_images
import png

def online_cut_patches(im, im_size, stride):
    """
    function for crop the image to subpatches, will include corner cases
    the return position (x,y) is the up left corner of the image
    
    Args:
        im (np.ndarray): the image for cropping
        im_size (int): the sub-image size.
        stride (int): the pixels between two sub-images.
    
    Returns:
        (list, list): list of image reference and list of its corresponding positions
    """
    im_list = []
    position_list = []

    h, w, _ = im.shape
    if h < im_size:
        h_ = np.array([0])
    else:
        h_ = np.arange(0, h - im_size + 1, stride)
        if h % stride != 0:
            h_ = np.append(h_, h-im_size)

    if w < im_size:
        w_ = np.array([0])
    else:
        w_ = np.arange(0, w - im_size + 1, stride)
        if w % stride != 0:
            w_ = np.append(w_, w - im_size)

    for i in h_:
        for j in w_:   	
            temp = np.uint8(im[i:i+im_size,j:j+im_size,:])
            im_list.append(temp)
            position_list.append((i,j))
    return im_list, position_list


def prepare_glas(side_length: int, stride: int, scales: List[int], network_image_size: int) -> None:
    """
    crop the training images and rename it with project convention
    e.g. imageName-[tumor, stroma, normal].png
    the image will be resized to network image size

    Args:
        side_length (int): the crop image length
        stride (int): the steps for cutting a new image
    """
    
    image_path = 'Dataset_glas/1.training/origin_ims'
    mask_path = 'Dataset_glas/1.training/mask'
    destination = 'Dataset_glas/1.training/img' # the output directory of the cropped images
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    training_ims = os.listdir(image_path)
    summary = []

    print(f'start generating train images ...')
    for i in tqdm(range(1, len(training_ims)+1)):
        image_name = f'train_{i}.bmp'
        mask_name = f'train_{i}_anno.bmp'
        im = np.asarray(Image.open(os.path.join(image_path, image_name)))
        mask = np.asarray(Image.open(os.path.join(mask_path, mask_name)))
        crop_list, positions_list = online_cut_patches(im, side_length, stride)
        for j in range(len(crop_list)):
            crop_im = crop_list[j]
            position = positions_list[j]
            crop_mask = mask[position[0]:position[0]+side_length, position[1]:position[1]+side_length]
            has_tumor = 0
            has_normal = 0

            # if crop contains at least 5% pixels as corresponding class, we define it exists in the big label
            if np.sum(crop_mask > 0) / crop_mask.size > 0.05:
                has_tumor = 1
            if np.sum(crop_mask == 0) / crop_mask.size > 0.05:
                has_normal = 1
            Image.fromarray(crop_im).save(os.path.join(destination, f'train_{i}_{j}-[{has_normal}, {has_tumor}].png'))
            summary.append((has_normal, has_tumor))
    
    print(Counter(summary))
    print('finish processing training images!')
    print()
    print('start processing validation and test images...')

    validation_folder_name = 'glas_valid'
    validation_dataset_path = 'Dataset_glas/2.validation/img'
    if not os.path.exists(validation_folder_name):
        os.mkdir(validation_folder_name)

    print('crop validation set images ...')
    crop_validation_images(validation_dataset_path, network_image_size, network_image_size, scales, validation_folder_name)
    print('cropping finishes!')

    # process mask only change the bmp img to png and add the color palette for better visualization
    def process_mask(mask_folder_path, destination):
        for mask_name in os.listdir(mask_folder_path):
            mask = np.asarray(Image.open(os.path.join(mask_folder_path, mask_name))).copy()
            # this three steps, convert tumor to 0, background to 2
            mask[mask > 0] = 1
            palette = [(64, 128, 0), (0, 64, 128)]
            with open(os.path.join(destination, f'{mask_name.split(".")[0]}.png'), 'wb') as f:
                w = png.Writer(mask.shape[1], mask.shape[0], palette=palette, bitdepth=8)
                w.write(f, mask.astype(np.uint8))

    validation_mask_path = 'Dataset_glas/2.validation/origin_mask'
    destination = 'Dataset_glas/2.validation/mask'
    if not os.path.exists(destination):
        os.mkdir(destination)
    process_mask(validation_mask_path, destination)

    print('mask processing finished!')


if __name__ == "__main__":
    side_length = 112
    stride = 56
    network_image_size = 224
    scales = [1, 1.25, 1.5, 1.75, 2]

    prepare_glas(side_length, stride, scales, network_image_size)