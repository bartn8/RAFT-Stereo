import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os

DEVICE = 'cpu'

def pil2cv(image: Image) -> np.ndarray:
    mode = image.mode
    print(f"image mode: {mode}")
    new_image: np.ndarray
    if mode == '1':
        new_image = np.array(image, dtype=np.uint8)
        new_image *= 255
    elif mode == 'L':
        new_image = np.array(image, dtype=np.uint8)
    elif mode == 'LA' or mode == 'La':
        new_image = np.array(image.convert('RGBA'), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif mode == 'RGB':
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif mode == 'RGBA':
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif mode == 'LAB':
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_LAB2BGR)
    elif mode == 'HSV':
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
    elif mode == 'YCbCr':
        # XXX: not sure if YCbCr == YCrCb
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_YCrCb2BGR)
    elif mode == 'P' or mode == 'CMYK':
        new_image = np.array(image.convert('RGB'), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif mode == 'PA' or mode == 'Pa':
        new_image = np.array(image.convert('RGBA'), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f'unhandled image color mode: {mode}')

    return new_image

def load_image(imfile):
    img = np.array(pil2cv(Image.open(imfile))).astype(np.uint8)
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        clean_left_images = sorted(glob.glob(args.clean_left_imgs, recursive=True))
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile0, imfile1, imfile2) in tqdm(list(zip(clean_left_images, left_images, right_images))):
            image0 = load_image(imfile0)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image0, image1, image2 = padder.pad(image0, image1, image2)

            _, flow_up = model(image0, image1, image2, iters=args.valid_iters, test_mode=True)
            file_stem = imfile1.split('/')[-1]
            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')

            flow_pr = padder.unpad(flow_up.float()).cpu().squeeze(0)

            dmap = flow_pr.cpu().numpy().squeeze()
            dmap = -dmap
            print(f"min: {np.min(dmap)}, max: {np.max(dmap)}")
            #dmap = (dmap-np.mean(dmap))/np.std(dmap)
            #print(f"min: {np.min(dmap)}, max: {np.max(dmap)}")
            #dmap = dmap * -255
            dmap = np.clip(dmap, 0, 255)
            dmap16 = ((dmap / 255) * 65535.0).astype(np.uint16)
            dmap = dmap.astype(np.uint8)
            dmaprgb = cv2.applyColorMap(dmap, cv2.COLORMAP_JET)
        
            cv2.imwrite(str(output_directory / f"{file_stem}_jet.png"), dmaprgb)
            cv2.imwrite(str(output_directory / f"{file_stem}_depth16.png"), dmap16, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-c', '--clean_left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
