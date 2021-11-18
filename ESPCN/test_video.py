from train import ESPCN
from utils import AverageMeter, calc_psnr
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os

def test_video(model_path, video_path, upscale_factor, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}\n")
    
    model = ESPCN(upscale_factor)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    hr_video = cv2.VideoCapture(video_path)

    video_fps = hr_video.get(cv2.CAP_PROP_FPS)
    frame_count = int(hr_video.get(cv2.CAP_PROP_FRAME_COUNT))

    hr_width = int(hr_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    hr_height = int(hr_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    sr_writer = cv2.VideoWriter(os.path.join(output_path, os.path.splitext(os.path.basename(video_path))[0] + f"_espcn_x{upscale_factor}.avi"), cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, (hr_width, hr_height))
    bicubic_writer = cv2.VideoWriter(os.path.join(output_path, os.path.splitext(os.path.basename(video_path))[0] + f"_bicubic_x{upscale_factor}.avi"), cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, (hr_width, hr_height))

    lr_width = hr_width // upscale_factor
    lr_height = hr_height // upscale_factor

    psnr_meter = AverageMeter()
    progress_bar = tqdm(range(frame_count), ncols=100)
    progress_bar.set_description(f'ESPCN SR')
    for i in progress_bar:
        _, sample = hr_video.read()

        hr_image = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
        lr_image = hr_image.resize((lr_width, lr_height), Image.BICUBIC)
        bicubic_image = lr_image.resize((hr_width, hr_height), Image.BICUBIC)

        hr_image = np.array(hr_image).astype(np.float32)
        lr_image = np.array(lr_image).astype(np.float32)
        bicubic_image = np.array(bicubic_image).astype(np.float32)

        hr_image_ycrcb = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
        lr_image_ycrcb = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
        bicubic_image_ycrcb = cv2.cvtColor(bicubic_image, cv2.COLOR_RGB2YCrCb)

        hr_y = hr_image_ycrcb[:,:,0]
        lr_y = lr_image_ycrcb[:,:,0]

        hr_y /= 255.
        lr_y /= 255.
        bicubic_image /= 255.

        hr_y = torch.from_numpy(hr_y).to(device)
        hr_y = hr_y.unsqueeze(0).unsqueeze(0)

        lr_y = torch.from_numpy(lr_y).to(device)
        lr_y = lr_y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            predicted = model(lr_y)

        psnr = calc_psnr(predicted, hr_y)
        psnr_meter.update(psnr, 1)
        print(f'\nPSNR: {psnr:.2f}')

        predicted = predicted.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        sr_output = np.array([predicted, bicubic_image_ycrcb[..., 1], bicubic_image_ycrcb[..., 2]]).transpose([1, 2, 0])
        sr_output = np.clip(cv2.cvtColor(sr_output, cv2.COLOR_YCrCb2BGR), 0.0, 255.0).astype(np.uint8)
        sr_writer.write(sr_output)

        bicubic_output = cv2.cvtColor(bicubic_image_ycrcb, cv2.COLOR_YCrCb2BGR).astype(np.uint8)
        bicubic_writer.write(bicubic_output)

    print(f'average PSNR: {psnr_meter.avg:.2f}')


if __name__ == '__main__':
    model_path = './assets/models/best.pth'
    video_path = './dataset/video/Bosphorus_1920x1080_30fps_420_8bit_AVC_MP4.mp4'
    output_path = './assets/outputs'
    upscale_factor = 3

    test_video(model_path, video_path, upscale_factor, output_path)
