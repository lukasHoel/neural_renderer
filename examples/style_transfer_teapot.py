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
from vgg_loss import GatysVggLoss

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
        renderer = nr.Renderer(camera_mode='look_at', image_size=image_size, perspective=False)
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer

        # setup content and style losses
        self.vgg_loss = GatysVggLoss(vgg_path)
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_tv = lambda_tv

        # setup graphics projection matrix
        self.P = None

    def forward(self, calc_loss=True):
        self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        image, _, mask = self.renderer(self.vertices, self.faces, torch.sigmoid(self.textures))  # [batch_size, RGB, image_size, image_size]

        if not calc_loss:
            return image

        s, c, tv = self.vgg_loss(image, None, self.style_image, mask)
        loss = self.lambda_content * c + self.lambda_style * s + self.lambda_tv * tv

        #loss = torch.nn.MSELoss()(image, self.style_image)

        """
        #print(f"C: {c}, S: {s}, TV: {tv}")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torchvision.transforms.ToPILImage()(image.squeeze().detach().cpu()))
        ax[1].imshow(mask.squeeze().cpu().detach().numpy())
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
    parser.add_argument('-io', '--mesh', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-is', '--filename_style', type=str, default="/home/hoellein/datasets/styles/3style/14-2.jpg")
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'style_transfer_teapot_result.gif'))
    parser.add_argument('-vgg', '--vgg_model_path', type=str, default="/home/hoellein/models/vgg_conv.pth")
    parser.add_argument('-lc', '--lambda_content', type=float, default=0)
    parser.add_argument('-ls', '--lambda_style', type=float, default=1)
    parser.add_argument('-ltv', '--lambda_tv', type=float, default=1e7)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-e', '--epochs', type=int, default=300)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    model = Model(args.mesh, args.filename_style, args.vgg_model_path,
                  lambda_content=args.lambda_content,
                  lambda_style=args.lambda_style,
                  lambda_tv=args.lambda_tv,
                  image_size=args.size)
    model.cuda()

    # optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loop = tqdm(range(args.epochs))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss, _ = model()
        loop.set_postfix({"loss": loss.detach().cpu().numpy().item()})
        loss.backward()
        optimizer.step()

    # draw object
    loop = tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
        images, _, _ = model.renderer(model.vertices, model.faces, torch.sigmoid(model.textures))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, image)
    make_gif(args.filename_output)
    nr.save_obj(args.filename_output + ".obj", model.vertices[0], model.faces[0], model.textures[0])


if __name__ == '__main__':
    main()
