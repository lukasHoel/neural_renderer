# taken from: https://github.com/krrish94/ENet-ScanNet/blob/master/data/scannet.py

import os
import cv2
import numpy as np
from PIL import Image
from os.path import join
from abstract_dataset import Abstract_Dataset


class ScanNetDataset(Abstract_Dataset):

    orig_sizes = {
        # in format (h, w)
        "rgb": (240, 320),
        "label": (240, 320),
        "uv": (480, 640)
    }

    stylized_images_sort_key = {
        "default": lambda x: int(x.split(".")[0]),
    }

    def __init__(self,
                 root_path,
                 transform_rgb=None,
                 resize=False,
                 resize_size=(256, 256),
                 cache=False,
                 verbose=False):
        Abstract_Dataset.__init__(self,
                                  root_path=root_path,
                                  transform_rgb=transform_rgb,
                                  resize=resize,
                                  resize_size=resize_size,
                                  cache=cache,
                                  verbose=verbose)

    def get_scenes(self):
        return os.listdir(self.root_path)

    def get_colors(self, scene_path, extensions=["jpg", "png"]):
        """
        Return absolute paths to all colors images for the scene (sorted!)
        """
        color_path = join(scene_path, "color")
        sort_key = ScanNetDataset.stylized_images_sort_key["default"]
        if not os.path.exists(color_path) or not os.path.isdir(color_path):
            return []

        colors = os.listdir(color_path)
        colors = [c for c in colors if any(c.endswith(x) for x in extensions)]
        colors = sorted(colors, key=sort_key)
        colors = [join(color_path, f) for f in colors]

        return colors

    def get_depth(self, scene_path):
        """
        Return absolute paths to all depth images for the scene (sorted!)
        """

        # load rendered opengl depth
        def load_rendered_depth(scene_path):
            uv_path = join(scene_path, "uv")
            if not os.path.exists(uv_path) or not os.path.isdir(uv_path):
                return []

            files = sorted(os.listdir(uv_path), key=lambda x: int(x.split(".")[0]))
            return [join(uv_path, f) for f in files if "npy" in f and 'depth' in f]
        rendered_depth_npy = load_rendered_depth(scene_path)

        # load original sensor depth
        depth_path = join(scene_path, "depth")
        if not os.path.exists(depth_path) or not os.path.isdir(depth_path):
            return []

        depth = sorted(os.listdir(depth_path), key=lambda x: int(x.split(".")[0]))
        depth = [join(depth_path, f) for f in depth]

        # choose opengl depth, if sensor depth not available
        if len(depth) == 0:
            self.rendered_depth = True
            return rendered_depth_npy
        else:
            self.rendered_depth = False
            return depth

    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
        """
        extrinsics_path = join(scene_path, "pose")

        if not os.path.exists(extrinsics_path) or not os.path.isdir(extrinsics_path):
            return []

        extrinsics = sorted(os.listdir(extrinsics_path), key=lambda x: int(x.split(".")[0]))
        extrinsics = [join(extrinsics_path, f) for f in extrinsics]

        return extrinsics

    def get_intrinsics(self, scene_path):
        """
        Return 3x3 numpy array as intrinsic K matrix for the scene and (W,H) image dimensions if available
        """
        intrinsics = np.identity(4, dtype=np.float32)
        w = 0
        h = 0
        file = [join(scene_path, f) for f in os.listdir(scene_path) if ".txt" in f]
        if len(file) == 1:
            file = file[0]
            self.intrinsics_file = file
            with open(file) as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    if "fx_color" in l:
                        fx = float(l.split(" = ")[1])
                        intrinsics[0,0] = fx
                    if "fy_color" in l:
                        fy = float(l.split(" = ")[1])
                        intrinsics[1,1] = fy
                    if "mx_color" in l:
                        mx = float(l.split(" = ")[1])
                        intrinsics[0,2] = mx
                    if "my_color" in l:
                        my = float(l.split(" = ")[1])
                        intrinsics[1,2] = my
                    if "colorWidth" in l:
                        w = int(l.split(" = ")[1])
                    if "colorHeight" in l:
                        h = int(l.split(" = ")[1])

        return intrinsics, (w,h)

    def get_uvs(self, scene_path):
        """
        Return absolute paths to all uvmap images for the scene (sorted!)
        """
        def load_folder(folder):
            if not os.path.exists(folder) or not os.path.isdir(folder):
                return [], []

            files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
            uvs_npy = [join(folder, f) for f in files if "npy" in f and not 'angle' in f and not 'depth' in f]
            uvs_png = [join(folder, f) for f in files if "png" in f and not 'angle' in f and not 'depth' in f]
            return uvs_npy, uvs_png

        uv_path = join(scene_path, "uv")
        uvs_npy, uvs_png = load_folder(uv_path)

        if len(uvs_npy) >= len(uvs_png):
            self.uv_npy = True
            return uvs_npy
        else:
            self.uv_npy = False
            return uvs_png

    def load_extrinsics(self, idx):
        """
        load the extrinsics item from self.extrinsics

        :param idx: the item to load

        :return: the extrinsics as numpy array
        """

        extrinsics = open(self.extrinsics[idx], "r").readlines()
        extrinsics = [[float(item) for item in line.split(" ")] for line in extrinsics]
        extrinsics = np.array(extrinsics, dtype=np.float32)

        return extrinsics

    def load_uvmap(self, idx, pyramid_idx=0):
        """
        load the uvmap item from self.uv_maps

        :param idx: the item to load

        :return: the uvmap as PIL image or numpy array
        """

        file = self.uv_maps[idx]
        if self.uv_npy:
            return np.load(file)
        else:
            return Image.open(file)

    def load_depth(self, idx):
        file = self.depth_images[idx]
        if not self.rendered_depth:
            d = np.asarray(Image.open(file)) / 1000.0
        else:
            d = np.load(file)
            d = d[:, :, :1]  # only keep first channel, but slice it instead of '[:,:,0]' to keep the dim

        return d

    def calculate_mask(self, uvmap, depth=None):
        """
        calculate the uvmap mask item from uvmap (valid values == 1)

        :param idx: the uvmap from which to calculate the mask

        :return: the mask as PIL image
        """

        mask = np.asarray(uvmap)
        if self.uv_npy:
            mask_bool = mask[:, :, 0] != 0
            mask_bool += mask[:, :, 1] != 0
            mask = mask_bool
        else:
            mask = mask[:, :, 2] == 0

        mask = Image.fromarray(mask)

        return mask
