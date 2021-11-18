from train import ESPCN
from PIL import Image
from utils import calc_psnr
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

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle('ESPCN Single Image Supre Resolution')
    fig.set_size_inches(20, 10, forward=True)

    hr_image = Image.open(image_path).convert('RGB')
    ax1.imshow(hr_image)
    ax1.set_title('HR image')
    
    lr_width = hr_image.width // upscale_factor
    lr_height = hr_image.height // upscale_factor
    hr_width = lr_width * upscale_factor
    hr_height = lr_height * upscale_factor

    hr_image = hr_image.resize((hr_width, hr_height), Image.BICUBIC)
    lr_image = hr_image.resize((lr_width, lr_height), Image.BICUBIC)
    
    bicubic_image = lr_image.resize((hr_width, hr_height), Image.BICUBIC)
    bicubic_image.save(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + f"_bicubic_x{upscale_factor}.png"))
    ax2.imshow(bicubic_image)
    ax2.set_title(f'BICUBIC image x{upscale_factor}')

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
    print(f'PSNR: {psnr:.2f}')

    predicted = predicted.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([predicted, bicubic_image_ycrcb[..., 1], bicubic_image_ycrcb[..., 2]]).transpose([1, 2, 0])
    output = np.clip(cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB), 0.0, 255.0).astype(np.uint8)
    output = Image.fromarray(output)
    output.save(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + f"_espcn_x{upscale_factor}.png"))
    
    ax3.imshow(output)
    ax3.set_title(f'SR image x{upscale_factor}  PSNR: {psnr:.2f}')
    fig.savefig(os.path.join(output_path, os.path.splitext(os.path.basename(image_path))[0] + f"_result_x{upscale_factor}.png"))
    plt.show()
    

if __name__ == '__main__':
    model_path = './assets/models/best.pth'
    image_path = "D:\\book\\娴娴\\壁纸\\娴娴8.png"
    output_path = './assets/outputs'
    upscale_factor = 3

    test_image(model_path, image_path, upscale_factor, output_path)
