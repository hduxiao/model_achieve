from torch.utils.data import DataLoader
from utils import AverageMeter, calc_psnr
from dataset import ImageTrainDataset, ImageValidDataset
from model import ESPCN
from tqdm import tqdm
import torch.nn as nn
import torch
import copy
import os

def train(model, data_loader, device, criterion, optimizer):
    model.train()
    loss_meter = AverageMeter()

    for data in data_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        predicted = model(inputs)
        loss = criterion(predicted, labels)
        loss_meter.update(loss.item(), len(predicted))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_meter

def evaluate(model, data_loader, device, criterion):
    model.eval()
    psnr_meter = AverageMeter()
    loss_meter = AverageMeter()

    for data in data_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predicted = model(inputs).clamp(0.0, 1.0)
            loss = criterion(predicted, labels)
            loss_meter.update(loss.item(), len(inputs))

        psnr_meter.update(calc_psnr(predicted, labels), len(inputs))

    return psnr_meter, loss_meter


if __name__ == '__main__':
    # set params
    train_set = './dataset/train'
    valid_set = './dataset/val'
    output_path = './assets/models'
    upscale_factor = 3
    learning_rate = 1e-3
    batch_size = 64
    epochs = 100
    seed = 4524

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}\n")

    # set seed for reproducibility
    torch.manual_seed(seed)

    # dataloader
    train_set = ImageTrainDataset(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, pin_memory=True)
    valid_set = ImageValidDataset(valid_set)
    # here the batch_size must set 1
    # because the shape of data in the same batch need to be consistent
    valid_loader = DataLoader(valid_set, batch_size=1, pin_memory=True)

    # model
    model = ESPCN(upscale_factor)
    model.to(device)

    # loss and optimizer
    # as per paper, the final layer, learns 10 times slowers
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([{'params': model.feature_map.parameters(), 'lr': learning_rate},
        {'params': model.sub_pixel_layer.parameters(), 'lr': learning_rate * 0.1}])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90], gamma=0.1)

    # train
    best_epoch = 0
    best_psnr = 0.0
    best_weight = copy.deepcopy(model.state_dict())

    progress_bar = tqdm(range(epochs), ncols=100)
    for epoch in progress_bar:
        progress_bar.set_description(f'epoch {epoch}')

        train_loss = train(model, train_loader, device, criterion, optimizer)
        psnr, valid_loss = evaluate(model, valid_loader, device, criterion)

        # save weight
        torch.save(model.state_dict(), os.path.join(output_path, '{:.2f}_epoch{}.pth'.format(psnr.avg, epoch)))
        # record
        if psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = psnr.avg
            best_weight = copy.deepcopy(model.state_dict())

        print(f'\nPSNR: {psnr.avg:.2f}, train loss: {train_loss.avg:.5f}, eval loss: {valid_loss.avg:.5f}\n')

    torch.save(best_weight, os.path.join(output_path, 'best.pth'))
    print(f'Best Epoch: {best_epoch}, psnr: {best_psnr:.2f}')
    
