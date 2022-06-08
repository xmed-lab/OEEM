import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import classification.network
import classification.dataset
from torch.utils.data import DataLoader
from classification.utils.metric import get_overall_valid_score
from classification.utils.generate_CAM import generate_validation_cam
from classification.utils.pyutils import crop_validation_images
from classification.utils.torchutils import PolyOptimizer
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=20, type=int)
    parser.add_argument('-epoch', default=20, type=int)
    parser.add_argument('-lr', default=0.01, type=float)
    parser.add_argument('-test_every', default=5, type=int, help="how often to test a model while training")
    parser.add_argument('-d','--device', nargs='+', help='GPU id to use parallel', required=True, type=int)
    parser.add_argument('-m', type=str, help='the save model name')
    parser.add_argument('-ckpt', type=str, help='the checkpoint model name')
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epoch
    base_lr = args.lr
    test_every = args.test_every
    devices = args.device
    model_name = args.m
    ckpt = args.ckpt

    with open('configuration.yml') as f:
        config = yaml.safe_load(f)
    mean = config['mean']
    std = config['std']
    num_class = config['num_class']
    network_image_size = config['network_image_size']
    scales = config['scales']

    if not os.path.exists('./classification/modelstates'):
        os.mkdir('./classification/modelstates')
    if not os.path.exists('/.classification/result'):
        os.mkdir('./classification/result')
    

    validation_folder_name = 'classification/glas_valid'
    validation_dataset_path = 'classification/glas/2.validation/img'
    validation_mask_path = 'classification/glas/2.validation/mask'
    if not os.path.exists(validation_folder_name):
        os.mkdir(validation_folder_name)
    print('crop validation set images ...')
    crop_validation_images(validation_dataset_path, network_image_size, network_image_size, scales, validation_folder_name)
    print('cropping finishes!')

    # EXCLUSIVELY FOR TRAINING
    if model_name == None:
        raise Exception("Model name is not provided for the traning phase!")
    # load model
    resnet38_path = "classification/weights/res38d.pth"
    net = classification.network.wideResNet()
    net.load_state_dict(torch.load(resnet38_path), strict=False)
    
    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    
    # data augmentation
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=network_image_size, scale=(0.7, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=mean, std=std)
    ])

    # load training dataset
    data_path_name = f'classification/glas/1.training/img'
    TrainDataset = classification.dataset.OriginPatchesDataset(data_path_name=data_path_name, transform=train_transform)
    print("train Dataset", len(TrainDataset))
    TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
    TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=4, sampler=TrainDatasampler, drop_last=True)

    # optimizer and loss
    optimizer = PolyOptimizer(net.parameters(), base_lr, weight_decay=1e-4, max_step=epochs, momentum=0.9)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
    regression_criteria = torch.nn.MSELoss(reduction='mean').cuda()
    criteria.cuda()

    # train loop
    loss_t = []
    iou_v = []
    best_val = 0
    
    for i in range(epochs):
        count = 0
        running_loss = 0.
        net.train()

        for img, label, area in tqdm(TrainDataloader):
            count += 1
            img = img.cuda()
            label = label.cuda()
            
            scores = net(img)
            loss = criteria(scores, label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / count
        loss_t.append(train_loss)

        valid_iou = 0
        if test_every != 0 and ((i + 1) % test_every == 0 or (i + 1) == epochs):
            net_cam = classification.network.wideResNet_cam()
            pretrained = net.state_dict()
            pretrained = {k[7:]: v for k, v in pretrained.items()}
            pretrained['fc1.weight'] = pretrained['fc1.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
            if 'fcregression.weight' in pretrained: 
                del pretrained['fcregression.weight']
                del pretrained['fcregression.bias']
            net_cam.load_state_dict(pretrained)
            net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()

            # calculate MIOU
            valid_image_path = os.path.join(validation_folder_name, model_name)
            generate_validation_cam(net_cam, config, batch_size, validation_dataset_path, validation_folder_name, model_name)
            valid_iou = get_overall_valid_score(valid_image_path, validation_mask_path, num_workers=8)
            iou_v.append(valid_iou)
            
            if valid_iou > best_val:
                print("Updating the best model..........................................")
                best_val = valid_iou
                torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./classification/modelstates/" + model_name + "_best.pth")
        
        print(f'Epoch [{i+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid mIOU: {valid_iou:.4f}, Valid Dice: {2 * valid_iou / (1 + valid_iou):.4f}')

    torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "./classification/modelstates/" + "_" + model_name + "_last.pth")

    plt.figure(1)
    plt.plot(loss_t)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('train loss')
    plt.savefig('./classification/result/train_loss.png')
    plt.close()

    plt.figure(2)
    plt.plot(iou_v)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('valid accuracy')
    plt.savefig('./classification/result/valid_iou.png')
