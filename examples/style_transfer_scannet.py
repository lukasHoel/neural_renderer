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
from pathlib import Path

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


class Model(nn.Module):
    def __init__(self, filename_obj, filename_style, vgg_path,
                 lambda_content=1, lambda_style=1, lambda_tv=0.01,
                 image_size=224):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj, normalization=False)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 4
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # load reference image
        style_image = torch.from_numpy(imread(filename_style).astype('float32') / 255.).permute(2,0,1)[None, ::]
        style_image = torch.nn.functional.interpolate(style_image, (image_size, image_size), mode='bilinear')
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

    def forward(self, content_img, extrinsics, intrinsics, mask, calc_loss=True):
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
        image = torch.flip(image, (2,))

        if not calc_loss:
            return image

        s, c, tv = self.vgg_loss(image, content_img, self.style_image, mask)
        loss = self.lambda_content * c + self.lambda_style * s + self.lambda_tv * tv

        """
        print(f"C: {c}, S: {s}, TV: {tv}")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(torchvision.transforms.ToPILImage()(content_img.squeeze()))
        ax[1].imshow(torchvision.transforms.ToPILImage()(image.squeeze()))
        ax[2].imshow(mask.cpu().numpy())
        plt.show()
        """

        return loss, image


def make_gif(filename, pattern='/tmp/_tmp_*.png', remove=True):
    with imageio.get_writer(filename, mode='I') as writer:
        files = glob.glob(pattern)
        files = sorted(files, key=lambda x: int(x.split(".")[-2].split("_")[-1]))
        for filename in files:
            writer.append_data(imageio.imread(filename))
            if remove:
                os.remove(filename)
    writer.close()


def get_key(args):
    style = args.filename_style
    style = os.path.basename(style)

    key = f"style_transfer_scannet_{args.scene}_{style}_content_{args.lambda_content}_style_{args.lambda_style}_tv_{args.lambda_tv}" \
          f"_epochs_{args.epochs}_lr_{args.learning_rate}_images_{args.max_images}_repeat_{args.index_repeat}"

    dir = os.path.join(args.out_dir, args.scene, style)

    return dir, key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default="/home/hoellein/datasets/scannet/train/")
    parser.add_argument('-if', '--image_folder', type=str, default="_additional")
    parser.add_argument('-s', '--scene', type=str, default="scene0673_00")
    parser.add_argument('-is', '--filename_style', type=str, default="/home/hoellein/datasets/styles/3style/14-2.jpg")
    parser.add_argument('-d', '--out_dir', type=str, default=os.path.join(data_dir))
    parser.add_argument('-vgg', '--vgg_model_path', type=str, default="/home/hoellein/models/vgg_conv.pth")
    parser.add_argument('-lc', '--lambda_content', type=float, default=1e2)
    parser.add_argument('-ls', '--lambda_style', type=float, default=1)
    parser.add_argument('-ltv', '--lambda_tv', type=float, default=1e8)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-ln', '--log_nth', type=int, default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-mi', '--max_images', type=int, default=-1)
    parser.add_argument('-ir', '--index_repeat', type=int, default=1)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # construct out path
    dir, key = get_key(args)
    Path(dir).mkdir(parents=True, exist_ok=True)
    out_prefix = os.path.join(dir, key)

    # build dataset
    image_path = os.path.join(args.data_path, args.image_folder)
    d = ScanNet_Single_House_Dataset(root_path=image_path,
                                     scene=args.scene,
                                     verbose=True,
                                     transform_rgb=torchvision.transforms.ToTensor(),
                                     resize=True,
                                     resize_size=(args.size, args.size),
                                     max_images=1000,
                                     min_images=1)

    # construct model with mesh, style image and vgg
    mesh_name = f'{args.scene}_vh_clean_decimate_500000_uvs_blender.obj'
    mesh_path = os.path.join(args.data_path, 'scans', args.scene, mesh_name)
    model = Model(mesh_path, args.filename_style, args.vgg_model_path,
                  lambda_content=args.lambda_content,
                  lambda_style=args.lambda_style,
                  lambda_tv=args.lambda_tv,
                  image_size=args.size)
    model.cuda()

    # optimize textured mesh
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        # set up logging and saving
        items = tqdm(d)
        items.set_description(f"Epoch {epoch}")
        out_prefix_epoch = f'{out_prefix}_epoch_{epoch}'

        # optimize one epoch
        for i, (rgb, extrinsics, intrinsics, mask) in enumerate(tqdm(d)):
            # break epoch after this many images
            if -1 < args.max_images <= i:
                break

            # optimize each image this many times
            for _ in range(args.index_repeat):
                optimizer.zero_grad()
                loss, image = model(rgb.unsqueeze(0).cuda(), extrinsics, intrinsics, mask.cuda())
                loss.backward()
                items.set_postfix({"loss": loss.cpu().detach().numpy().item()})
                optimizer.step()

            # save optimized image directly (not after the epoch is done, this is faster)
            if epoch % args.log_nth == 0 or args.log_nth == -1:
                image = image.detach().cpu().numpy()[0].transpose((1, 2, 0))
                mask3 = torch.stack([mask, mask, mask], dim=0)  # from (H x W) to (3 x H x W)
                image_masked = np.where(mask3.detach().cpu().permute(1,2,0).numpy(), image, np.zeros_like(image))
                imsave(f'{out_prefix_epoch}_{i}.png', image_masked)

        # after each epoch, we collect the images and make a .gif out of them
        if epoch % args.log_nth == 0 or args.log_nth == -1:
            make_gif(f"{out_prefix_epoch}.gif", f"{out_prefix_epoch}*.png", remove=False)


if __name__ == '__main__':
    main()
