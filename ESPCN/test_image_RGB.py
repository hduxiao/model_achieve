from train_RGB import ESPCN
from PIL import Image
from utils_RGB import calc_psnr
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def test_image(model_path, image_path, upscale_factor, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}\n")
    
    model = ESPCN(upscale_factor)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    hr_image = cv2.imread(image_path)
    # calc width, height
    lr_height = hr_image.shape[0] // upscale_factor
    lr_width = hr_image.shape[1] // upscale_factor
    hr_width = lr_width * upscale_factor
    hr_height = lr_height * upscale_factor
    # resize
    hr_image = cv2.resize(hr_image, (hr_width, hr_height), interpolation = cv2.INTER_CUBIC)
    lr_image = cv2.resize(hr_image, (lr_width, lr_height), interpolation = cv2.INTER_CUBIC)
    hr_image = hr_image.astype(np.float32)
    lr_image = lr_image.astype(np.float32)
    hr_image /= 255.0
    lr_image /= 255.0
    lr_image = torch.from_numpy(lr_image)
    hr_image = torch.from_numpy(hr_image)
    lr_image = lr_image.permute(2, 0, 1)
    hr_image = hr_image.permute(2, 0, 1)
    
    hr_image = hr_image.unsqueeze(0)
    lr_image = lr_image.unsqueeze(0)

    print(hr_image.shape)
    print(lr_image.shape)

    with torch.no_grad():
        lr_image = lr_image.to(device)
        predicted = model(lr_image)

    # psnr = calc_psnr(predicted, hr_image)
    # print(f'PSNR: {psnr:.2f}')

    predicted = predicted.mul(255.0).cpu().numpy().squeeze(0)
    print(predicted.shape)

    isWritten = cv2.imwrite("C:\\Users\\hduxi\\Desktop\\test.png", predicted)

if __name__ == '__main__':
    model_path = './assets/models/best.pth'
    image_path = "C:\\Users\\hduxi\\Desktop\\ESPCN\\dataset\\test\\baboon.png"
    output_path = './assets/outputs'
    upscale_factor = 3

    test_image(model_path, image_path, upscale_factor, output_path)
