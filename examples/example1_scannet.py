"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import argparse

import torch
import numpy as np
from tqdm.auto import tqdm
import imageio
import torchvision

import neural_renderer as nr
from scannet_single_scene_dataset import ScanNet_Single_House_Dataset

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--mesh', type=str,
                        default="/home/hoellein/datasets/scannet/train/scans/scene0673_00/scene0673_00_vh_clean_decimate_500000_uvs_blender.obj")
    parser.add_argument('-s', '--scene', type=str, default="scene0673_00")
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    d = ScanNet_Single_House_Dataset(root_path="/home/hoellein/datasets/scannet/train/images",
                                     scene=args.scene,
                                     verbose=True,
                                     transform_rgb=torchvision.transforms.ToTensor(),
                                     load_uvs=True,
                                     load_uv_pyramid=False,
                                     create_instance_map=False,
                                     crop=False,
                                     resize=True,
                                     resize_size=(256,256),
                                     max_images=1000,
                                     min_images=1)

    # other settings
    texture_size = 2

    # load .obj
    vertices, faces = nr.load_obj(args.mesh, normalization=False)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu

    # create renderer
    renderer = nr.Renderer(camera_mode='projection', image_size=256)

    # draw object
    loop = tqdm(range(0, 360, 4))
    writer = imageio.get_writer(args.filename_output, mode='I')
    P = None
    for rgb, extrinsics, intrinsics, mask in tqdm(d):
        loop.set_description('Drawing')

        # get R, t, look from extrinsics
        extrinsics = extrinsics.cuda()
        R = extrinsics[None, :3, :3]
        t = extrinsics[None, :3, 3:].permute(0, 2, 1)
        look = extrinsics[:3, 2]

        if P is None:
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

        images, _, _ = renderer(vertices, faces, textures, K=P, R=R, t=t, at=look)  # [batch_size, RGB, image_size, image_size]
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        image = np.flipud(image)

        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(torchvision.transforms.ToPILImage()(rgb))
        ax[1].imshow(image)
        ax[2].imshow(mask.cpu().numpy())
        plt.show()
        """

        writer.append_data((255*image).astype(np.uint8))
    writer.close()


if __name__ == '__main__':
    main()
