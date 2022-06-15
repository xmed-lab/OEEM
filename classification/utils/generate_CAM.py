import json
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import dataset
import torch
import os


def generate_validation_cam(net, config, batch_size, dataset_path, validation_folder_name, model_name, elimate_noise=False, label_path=None):
    """
    Generate the class activation map for the validation set and evaluate.

    Args:
        net (torch.model): the classification model
        config (dict): configs from configuration.yml
        batch_size (int): batch to process the cam
        dataset_path (str): the address of the image dataset
        folder_name (str): the folder to store the cam output
        model_name (str): the name for this cam_output model
        elimate_noise (bool, optional): use image-level label to cancel some of the noise. Defaults to False.
        label_path (str, optional): if `eliminate_noise` is True, input the labels path. Defaults to None.
    """
    side_length = config['network_image_size']
    mean = config['mean']
    std = config['std']
    network_image_size = config['network_image_size']
    scales = config['scales']

    net.cuda()
    net.eval()

    crop_image_path = f'{validation_folder_name}/crop_images/'
    image_name_list = os.listdir(crop_image_path)
    extension_name = os.listdir(dataset_path)[0].split('.')[-1]
    
    for image_name in tqdm(image_name_list):
        orig_img = np.asarray(Image.open(f'{dataset_path}/{image_name}.{extension_name}'))
        w, h, _ = orig_img.shape
        ensemble_cam = np.zeros((2, w, h))
        
        for scale in scales:
            image_per_scale_path = crop_image_path + image_name + '/' + str(scale)
            scale = float(scale)
            offlineDataset = dataset.OfflineDataset(image_per_scale_path, transform=transforms.Compose([
                    transforms.Resize((network_image_size, network_image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            )
            offlineDataloader = DataLoader(offlineDataset, batch_size=batch_size, drop_last=False)

            w_ = int(w*scale)
            h_ = int(h*scale)
            interpolatex = side_length
            interpolatey = side_length
            if w_ < side_length:
                interpolatex = w_
            if h_ < side_length:
                interpolatey = h_

            with torch.no_grad():
                cam_list = []
                position_list = []
                for ims, positions in offlineDataloader:
                    cam_scores = net.foward_cam(ims.cuda())
                    cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False).detach().cpu().numpy()
                    cam_list.append(cam_scores)
                    position_list.append(positions.numpy())
                cam_list = np.concatenate(cam_list)
                position_list = np.concatenate(position_list)
                sum_cam = np.zeros((2, w_, h_))
                sum_counter = np.zeros_like(sum_cam)
                
                for k in range(cam_list.shape[0]):
                    y, x = position_list[k][0], position_list[k][1]
                    crop = cam_list[k]
                    sum_cam[:, y:y+side_length, x:x+side_length] += crop
                    sum_counter[:, y:y+side_length, x:x+side_length] += 1
                sum_counter[sum_counter < 1] = 1
                norm_cam = sum_cam / sum_counter
                norm_cam = F.interpolate(torch.unsqueeze(torch.tensor(norm_cam),0), (w, h), mode='bilinear', align_corners=False).detach().cpu().numpy()[0]

                # Use the image-level label to eliminate impossible pixel classes
                ensemble_cam += norm_cam                

        if elimate_noise:
            with open(f'{validation_folder_name}/{label_path}') as f:
                big_labels = json.load(f)
            big_label = big_labels[f'{image_name}.png']        
            for k in range(2):
                if big_label[k] == 0:
                    ensemble_cam[k, :, :] = -np.inf
                    
        result_label = ensemble_cam.argmax(axis=0)
        if not os.path.exists(f'{validation_folder_name}/{model_name}'):
            os.mkdir(f'{validation_folder_name}/{model_name}')
        np.save(f'{validation_folder_name}/{model_name}/{image_name}.npy', result_label)