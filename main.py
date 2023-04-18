from utils.utils import seed_everything
import argparse
#import wandb
from core.function import train, valid
from core.criterion import MultiClassDiceCELoss
import pickle
import numpy as np
from core.optimizer import CosineAnnealingWarmUpRestarts
from data.dataset import ImageDataset
from timm.models.layers import to_2tuple
from torch.utils.data import DataLoader
from models.ResUnet_A import ResUnetA
import torch
import torch.nn as nn
import os
import albumentations
import albumentations.pytorch
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
                    default=304, help='random seed')
parser.add_argument('--csv_path', type=str, required=True, metavar="FILE", help='Train and Valid Csv file directory path')
parser.add_argument('--log_path', type=str, default='./log', help='Save log file path')
parser.add_argument('--img_size', default=256, type=int, help='Resize img size')
parser.add_argument('--gpu', default="0", type=str, help='use gpu num')
parser.add_argument('--workers', default=1, type=int, help='dataloader workers')
parser.add_argument('--epoch', default=100, type=int, help='Train Epoch')
parser.add_argument('--batch_size', default=4, type=int, help='Train Batch size')
parser.add_argument('--in_channels', default=3, type=int, help='input image channels')
parser.add_argument('--out_channels', default=1, type=int, help='Number of class')
parser.add_argument('--filter_size', default=[32, 64, 128, 256, 512], help='ResUnet Filter size')
parser.add_argument('--optim_eps', default=1e-8, type=float, help='AdamW Optimizer eps parameter')
parser.add_argument('--optim_betas', default=(0.9, 0.999), help='AdamW Optimizer betas parameter')
parser.add_argument('--lr', default=1e-4, type=float, help='Optimizer learning rate parameter')
parser.add_argument('--weight_decay', default=0.95, type=float, help='Optimizer weight decay parameter')
parser.add_argument('--model_path', default=None, type=str, help='retrain model load path')
parser.add_argument('--model_save_path', default='./weight', type=str, help='model save path')
parser.add_argument('--write_iter_num', default=20, type=int, help='write learning situation iteration time')

def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Reading Data")
    train_csv = pd.read_csv(os.path.join(args.csv_path, 'Train_data.csv'))
    valid_csv = pd.read_csv(os.path.join(args.csv_path, 'Valid_data.csv'))
    
    image_shape = to_2tuple(arg.img_size)
    height, width = image_shape
    train_transform = albumentations.Compose(
            [
                albumentations.Resize(height, width, interpolation=cv2.INTER_LINEAR),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.ShiftScaleRotate(p=1, rotate_limit=90),
                    albumentations.VerticalFlip(p=1),
                    albumentations.RandomBrightnessContrast(p=1),
                    albumentations.GaussNoise(p=1)                    
                ],p=1),
                albumentations.Normalize(),
                albumentations.ToTensorV2()
            ]
        )
    valid_transform = albumentations.Compose(
            [
                albumentations.Resize(height, width, interpolation=cv2.INTER_LINEAR),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.VerticalFlip(p=1)
                ],p=1),
                albumentations.Normalize(),
                albumentations.ToTensorV2()
            ]
        )
        
    #del data_file
    train_dataset = ImageDataset(Image_path=train_csv, transform=None)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=True)
    valid_dataset = ImageDataset(Image_path=valid_csv, transform=None)
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=False)
    
    #model init
    print("Model Init")
    model = ResUnetA(img_size=args.img_size, channels=args.in_channels, classes=args.out_channels, filtersize=args.filter_size, check_sigmoid=False)
    optimizer = torch.optim.AdamW(model.parameters(), eps=args.optim_eps, betas=args.optim_betas,
                                lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.scheduler_t, T_mult=args.scheduler_tmult,
                                                    eta_max=args.scheduler_eta, T_up=args.scheduler_up, gamma=args.scheduler_gamma)
    criterion = MultiClassDiceCELoss(num_classes=args.out_channels)

    model = model.to(device)
    criterion = criterion.to(device)    
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location={'cuda:0':'cpu'})
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best_loss = 0
    for epoch in range(start_epoch, args.epoch):
        is_best = False
        file = open(os.path.join(args.log_path, f'{epoch}_log.txt'), 'a')
        if args.distributed:
            train_sampler.set_epoch(epoch)    
        train(model=model, write_iter_num=args.write_iter_num, train_dataset=trainloader, optimizer=optimizer, 
                    device=device, criterion=criterion, epoch=epoch, file=file)
        accuracy = valid(model=model, write_iter_num=args.write_iter_num, valid_dataset=validloader, criterion=criterion, 
                               device=device, epoch=epoch, file=file)
        scheduler.step()
        is_best = accuracy > best_loss
        best_loss = max(best_loss, accuracy)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_loss,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best=is_best, path=args.model_save_path)
        file.close()
    
if __name__ == '__main__':
    main()