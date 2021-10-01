"""
Example 3. Texture-only style transfer.
"""
from __future__ import division
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms
from skimage.io import imread, imsave
from tqdm.auto import tqdm
import imageio

import neural_renderer as nr
from scannet_single_scene_dataset import ScanNet_Single_House_Dataset
from vgg_loss import GatysVggLoss

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


class Model(nn.Module):
    def __init__(self, filename_obj, filename_style, vgg_path,
                 lambda_content=1, lambda_style=1, lambda_tv=0.01,
                 image_size=256):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj, normalization=False)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 4
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # load reference image
        style_image = torch.from_numpy(imread(filename_style).astype('float32') / 255.).permute(2,0,1)[None, ::]
        self.register_buffer('style_image', style_image)

        # setup renderer
        renderer = nr.Renderer(camera_mode='projection', image_size=image_size)
        self.renderer = renderer

        # setup content and style losses
        self.vgg_loss = GatysVggLoss(vgg_path)
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_tv = lambda_tv

        # setup graphics projection matrix
        self.P = None

    def forward(self, content_img, extrinsics, intrinsics, mask):
        # get R, t, look from extrinsics
        extrinsics = extrinsics.cuda()
        R = extrinsics[None, :3, :3]
        t = extrinsics[None, :3, 3:].permute(0, 2, 1)
        look = extrinsics[:3, 2]

        if self.P is None:
            # construct graphics projection matrix (only once, is shared across all views)
            intrinsics = intrinsics.cuda()
            K = intrinsics[None, :3, :3]
            width = 1296
            height = 968
            n = 0.1
            f = 100.0
            P = torch.zeros(1, 4, 4).type_as(K)
            P[:, 0, 0] = 2 * K[:, 0, 0] / width
            P[:, 1, 1] = 2 * K[:, 1, 1] / height
            P[:, 0, 2] = -(2 * K[:, 0, 2] / width - 1)
            P[:, 1, 2] = -(2 * K[:, 1, 2] / height - 1)
            P[:, 2, 2] = -(f + n) / (f - n)
            P[:, 2, 3] = -2 * f * n / (f - n)
            P[:, 3, 2] = -1
            self.P = P

        image, _, _ = self.renderer(self.vertices, self.faces, torch.sigmoid(self.textures), K=self.P, R=R, t=t, at=look)  # [batch_size, RGB, image_size, image_size]
        c, s, tv = self.vgg_loss(image, content_img, self.style_image, mask)
        loss = self.lambda_content * c + self.lambda_style * s + self.lambda_tv * tv

        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(torchvision.transforms.ToPILImage()(content_img.squeeze()))
        ax[1].imshow(torchvision.transforms.ToPILImage()(image.squeeze()))
        ax[2].imshow(mask.cpu().numpy())
        plt.show()
        """

        return loss, image


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--mesh', type=str, default="/home/hoellein/datasets/scannet/train/scans/scene0673_00/scene0673_00_vh_clean_decimate_500000_uvs_blender.obj")
    parser.add_argument('-s', '--scene', type=str, default="scene0673_00")
    parser.add_argument('-is', '--filename_style', type=str, default="/home/hoellein/datasets/styles/3style/14-2.jpg")
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example3_scannet_result.gif'))
    parser.add_argument('-vgg', '--vgg_model_path', type=str, default="/home/hoellein/models/vgg_conv.pth")
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    d = ScanNet_Single_House_Dataset(root_path="/home/hoellein/datasets/scannet/train/images",
                                     scene=args.scene,
                                     verbose=True,
                                     transform_rgb=torchvision.transforms.ToTensor(),
                                     load_uvs=True,
                                     resize=True,
                                     resize_size=(args.size, args.size),
                                     max_images=1000,
                                     min_images=1)

    model = Model(args.mesh, args.filename_style, args.vgg_model_path, image_size=args.size)
    model.cuda()

    # optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5,0.999))
    for rgb, extrinsics, intrinsics, mask in tqdm(d):
        optimizer.zero_grad()
        loss, image = model(rgb.unsqueeze(0).cuda(), extrinsics, intrinsics, mask.cuda())
        loss.backward()
        optimizer.step()

    # draw object
    for i, (rgb, extrinsics, intrinsics, mask) in enumerate(tqdm(d)):
        image = model(rgb.unsqueeze(0).cuda(), extrinsics, intrinsics, mask.cuda())
        image = image.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % i, image)
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
