import numpy as np
from PIL import Image
import os
from collections import Counter
from os.path import join as osp
from tqdm import tqdm
from cv2 import imread
import cv2
import copy
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import png

def online_cut_patches(im, im_size, stride):
    """
    function for crop the image to subpatches, will include corner cases
    the return position (x,y) is the up left corner of the image
    Args:
        im (np.ndarray): the image for cropping
        im_size (int, optional): the sub-image size.
        stride (int, optional): the pixels between two sub-images.
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
            temp = Image.fromarray(np.uint8(im[i:i+im_size,j:j+im_size,:].copy()))
            im_list.append(temp)
            position_list.append((i,j))
    return im_list, position_list


def multiscale_online_crop(im, im_size, stride, scales):
    """
    First resize the image to different scales, then crop according to `im_size`

    Returns:
        scale_im_list: the image list
        scale_position_list: the images position
    """
    im = Image.fromarray(im)
    w, h = im.size
    scale_im_list = []
    scale_position_list = []
    for scale in scales:
        scaled_im = np.asarray(im.resize((int(w*scale), int(h*scale))))
        im_list, position_list = online_cut_patches(scaled_im, im_size, stride)
        scale_im_list.append(im_list)
        scale_position_list.append(position_list)

    return scale_im_list, scale_position_list

def get_average_image_size(path):
    """
    get the average size of the images in the path directory

    Args:
        path (str): image path
    """
    images = os.listdir(path)
    height = 0
    width = 0
    for image in images:
        h, w, _ = imread(path + '/' + image).shape
        height += h
        width += w

    print(height//len(images), width//len(images))
    return height//len(images), width//len(images)

def chunks(lst, num_workers=None, n=None):
    """
    a helper function for seperate the list to chunks

    Args:
        lst (list): the target list
        num_workers (int, optional): Default is None. When num_workers are not None, the function divide the list into num_workers chunks
        n (int, optional): Default is None. When the n is not None, the function divide the list into n length chunks

    Returns:
        llis: a list of small chunk lists
    """
    chunk_list = []
    if num_workers is None and n is None:
        print("the function should at least pass one positional argument")
        exit()
    elif n == None:
        n = int(np.ceil(len(lst)/num_workers))
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list
    else:
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list

def crop_validation_images(dataset_path, side_length, stride, scales, validation_folder_name):
    """
    if the scales are not modified, this function can run only once.
    crop the validation images to reduce the validation time
    the output is in `validation_cam_folder_name/crop_images`
    images are stored according to the image name

    Args:
        dataset_path (str): the validation dataset path
        side_length (int): the crop size
        stride (int): the distance between two crops
        scales (list): a list of scales to crop
        validation_cam_folder_name (str): the destination to store the validation cam
    """
    images = os.listdir(dataset_path)
    
    if not os.path.exists(f'{validation_folder_name}/crop_images'):
        os.mkdir(f'{validation_folder_name}/crop_images')
    
    for image in tqdm(images):
        if not os.path.exists(f'{validation_folder_name}/crop_images/{image.split(".")[0]}'):
            os.mkdir(f'{validation_folder_name}/crop_images/{image.split(".")[0]}')
        image_path = os.path.join(dataset_path, image)
        im = np.asarray(Image.open(image_path))
        scaled_im_list, scaled_position_list = multiscale_online_crop(im, side_length, stride, scales)
        
        for i in range(len(scales)):
            # create folder for each scales
            if not os.path.exists(f'{validation_folder_name}/crop_images/{image.split(".")[0]}/{scales[i]}'):
                os.mkdir(f'{validation_folder_name}/crop_images/{image.split(".")[0]}/{scales[i]}')
            
            for j in range(len(scaled_im_list[i])):
                scaled_im_list[i][j].save(f'{validation_folder_name}/crop_images/{image.split(".")[0]}/{scales[i]}/{scaled_position_list[i][j]}.png')

def predict_mask(image, threshold, minimal_size):
    """
    given the rgb image, get the foreground mask

    Args:
        image (PIL.image): the 
        threshold (int): (0, 255) the threshold for image in hsv format's Value channel
        minimal_size (int): the parts smaller than minimal_sized will be removed

    Returns:
        np.ndarray: the predicted mask, same shape as `image`, 0 is background, 1 is foreground 
    """
    # threshold
    image_t = np.array(image)
    temp = cv2.cvtColor(image_t, cv2.COLOR_RGB2HSV)
    threshold_saturation = threshold_otsu(temp[:,:,1])
    image_t[temp[:,:,1] < threshold_saturation] = [255, 255, 255]
    image_t[temp[:,:,1] >= threshold_saturation] = [0, 0, 0]
    image_t[temp[:,:,2] > threshold] = [255, 255, 255]
    image_t[temp[:,:,2] <= threshold] = [0, 0, 0]

    # erode
    erosion_size = 0
    erosion_shape = 2
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    image_e = cv2.erode(image_t, element)

    # connected components filtering
    image_c = copy.deepcopy(image_e)
    temp = cv2.cvtColor(image_e, cv2.COLOR_RGB2GRAY)
    temp[temp == 0] = 2  # prevent recognizing 0 as background
    label_image = label(temp)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area <= minimal_size:
            image_c[label_image == region.label] = [255, 255, 255] - image_c[label_image == region.label]

    # save the predict mask
    result = image_c[:, :, 1].astype(np.uint8)
    result[result == 0] = 1
    result[result == 255] = 0
    return result

def get_dataset_stats(dataset_root_path):
    imgs_path = osp(dataset_root_path, "origin_ims")
    img_h, img_w = get_average_image_size(imgs_path)
    means, stdevs = [], []
    img_list = []
    
    for file in os.listdir(imgs_path):
        img = Image.open(osp(imgs_path, file))
        img = np.asarray(img.resize((img_w, img_h))) # H, W, C
        img = img[:, :, :, None]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

def glas_join_crops_back(cropped_cam_path: str, origin_ims_path: str, side_length: int, stride: int, is_train: bool) -> None:
    """
    merge the cropped image mask to the original image size and save in the `cropped_cam_path` folder

    Args:
        cropped_cam_path (str): the predicted crop masks path
        origin_ims_path (str): the original image path
        side_length (int): the crop size
        stride (int): the step between crop images
        is_train 
    """
    partial_image_list = os.listdir(cropped_cam_path)
    # make a dict to tract wich images are in a group and should be merged back
    ims_dict = {}
    for partial_mask in partial_image_list:
        _, corresponding_im, index = partial_mask.split('_')
        index = int(index.split('-')[0])
        if is_train:
            if f'train_{corresponding_im}.bmp' not in ims_dict:
                ims_dict[f'train_{corresponding_im}.bmp'] = {}
            ims_dict[f'train_{corresponding_im}.bmp'][index] = os.path.join(cropped_cam_path, partial_mask)
        else:
            if f'{corresponding_im}.bmp' not in ims_dict:
                ims_dict[f'{corresponding_im}.bmp'] = {}
            ims_dict[f'{corresponding_im}.bmp'][index] = os.path.join(cropped_cam_path, partial_mask)

    # merge images to the size in validation set part
    for origin_im in os.listdir(origin_ims_path):
        im = np.asarray(Image.open(os.path.join(origin_ims_path, origin_im)))
        complete_mask = np.zeros((im.shape[0], im.shape[1]))
        sum_counter = np.zeros_like(complete_mask)
        _, position_list = online_cut_patches(im, im_size=side_length, stride=stride)

        for i in range(len(position_list)):
            partial_mask = np.load(ims_dict[origin_im][i], allow_pickle=True)
            position = position_list[i]
            complete_mask[position[0]:position[0]+side_length, position[1]:position[1]+side_length] += partial_mask
            sum_counter[position[0]:position[0]+side_length, position[1]:position[1]+side_length] += 1

        complete_mask = np.rint(complete_mask / sum_counter)
        palette = [(0, 64, 128), (243, 152, 0)]
        with open(os.path.join(cropped_cam_path, f'{origin_im.split(".")[0]}.png'), 'wb') as f:
            w = png.Writer(complete_mask.shape[1], complete_mask.shape[0], palette=palette, bitdepth=8)
            w.write(f, complete_mask.astype(np.uint8))

    # remove the cropped mask and only keep the complete mask
    for partial_image in partial_image_list:
        if os.path.exists(os.path.join(cropped_cam_path, partial_image)):
            os.remove(os.path.join(cropped_cam_path, partial_image))