import argparse
import os
from PIL import Image
import numpy as np
from scipy.stats.stats import mode
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from dataset import TrainingSetCAM
import network
from utils.pyutils import glas_join_crops_back
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=20, type=int)
    parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
    parser.add_argument('-ckpt', type=str, required=True, help='the checkpoint model name')
    args = parser.parse_args()

    batch_size = args.batch
    devices = args.device
    ckpt = args.ckpt

    with open('classification/configuration.yml') as f:
        data = yaml.safe_load(f)
    mean = data['mean']
    std = data['std']
    side_length = data['side_length']
    stride = data['stride']
    num_of_class = data['num_of_class']
    network_image_size = data['network_image_size']
    scales = data['scales']

    train_pseudo_mask_path = 'classification/' + ckpt.replace('.pth', '') + '_train_pseudo_mask'
    if not os.path.exists(train_pseudo_mask_path):
        os.mkdir(train_pseudo_mask_path)

    data_path_name = f'classification/glas/1.training/img'
    majority_vote = False
    
    dataset = TrainingSetCAM(data_path_name=data_path_name, transform=transforms.Compose([
                        transforms.Resize((network_image_size, network_image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                ]), patch_size=side_length, stride=stride, scales=scales, num_class=num_of_class
    )
    dataLoader = DataLoader(dataset, batch_size=1, drop_last=False)

    net_cam = network.wideResNet_cam(num_class=num_of_class)
    model_path = "classification/weights/" + ckpt + ".pth"
    pretrained = torch.load(model_path)['model']
    pretrained = {k[7:]: v for k, v in pretrained.items()}
    pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
    net_cam.load_state_dict(pretrained)

    net_cam.eval()
    net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()

    with torch.no_grad():
        for im_name, scaled_im_list, scaled_position_list, scales, big_label in tqdm(dataLoader):
            big_label = big_label[0]
            
            # training images have big labels, can be used to improve CAM performance
            eliminate_noise = True
            if len(big_label) == 1:
                eliminate_noise = False
            
            orig_img = np.asarray(Image.open(f'{data_path_name}/{im_name[0]}'))
            w, h, _ = orig_img.shape

            if majority_vote:
                ensemble_cam = []
            else:
                ensemble_cam = np.zeros((num_of_class, w, h))

            # get the prediction for each pixel in each scale
            for s in range(len(scales)):
                w_ = int(w*scales[s])
                h_ = int(h*scales[s])
                interpolatex = side_length
                interpolatey = side_length

                if w_ < side_length:
                    interpolatex = w_
                if h_ < side_length:
                    interpolatey = h_

                im_list = scaled_im_list[s]
                position_list = scaled_position_list[s]

                im_list = torch.vstack(im_list)
                im_list = torch.split(im_list, batch_size)

                cam_list = []
                for ims in im_list:
                    cam_scores = net_cam(ims.cuda())
                    cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear', align_corners=False).detach().cpu().numpy()
                    cam_list.append(cam_scores)
                cam_list = np.concatenate(cam_list)

                sum_cam = np.zeros((num_of_class, w_, h_))
                sum_counter = np.zeros_like(sum_cam)
            
                for k in range(cam_list.shape[0]):
                    y, x = position_list[k][0], position_list[k][1]
                    crop = cam_list[k]
                    sum_cam[:, y:y+side_length, x:x+side_length] += crop
                    sum_counter[:, y:y+side_length, x:x+side_length] += 1
                sum_counter[sum_counter < 1] = 1

                norm_cam = sum_cam / sum_counter
                norm_cam = F.interpolate(torch.unsqueeze(torch.tensor(norm_cam),0), (w, h), mode='bilinear', align_corners=False).detach().cpu().numpy()[0]
                
                # use the image-level label to eliminate impossible pixel classes
                if majority_vote:
                    if eliminate_noise:
                        for k in range(num_of_class):
                            if big_label[1-k] == 0:
                                norm_cam[k, :, :] = -np.inf
                
                    norm_cam = np.argmax(norm_cam, axis=0)        
                    ensemble_cam.append(norm_cam)
                else:
                    ensemble_cam += norm_cam
            
            if majority_vote:
                ensemble_cam = np.stack(ensemble_cam, axis=0)
                result_label = mode(ensemble_cam, axis=0)[0]
            else:
                if eliminate_noise:
                    for k in range(num_of_class):
                        if big_label[1-k] == 0:
                            ensemble_cam[k, :, :] = -np.inf
                            
                result_label = ensemble_cam.argmax(axis=0)
            
            np.save(f'{train_pseudo_mask_path}/{im_name[0].split(".")[0]}.npy', result_label)

    origin_ims_path = 'classification/glas/1.training/origin_ims'
    glas_join_crops_back(train_pseudo_mask_path, origin_ims_path, side_length, stride, is_train=True)
