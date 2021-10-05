import torchvision
import torch
import numpy as np
import os

from torch.utils.data import Dataset

from PIL import Image

from tqdm.auto import tqdm

import os.path
from os.path import join

from abc import ABC, abstractmethod


class Abstract_Dataset(Dataset, ABC):

    def __init__(self,
                 root_path,
                 transform_rgb=None,
                 resize=False,
                 resize_size=(256, 256),
                 cache=False,
                 verbose=False):
        # save all constructor arguments
        self.transform_rgb = transform_rgb
        self.resize = resize
        self.resize_size = resize_size
        if isinstance(resize_size, int):
            # self.resize_size = (resize_size, resize_size)
            pass
        self.verbose = verbose
        self.root_path = root_path
        self.use_cache = cache
        self.cache = {}

        # create data for this dataset
        self.create_data()

        if self.use_cache:
            print("Preloading all into cache")
            for i in tqdm(range(self.size)):
                self.__getitem__(i)
            print("Finished preloading")

    def create_data(self):
        self.rgb_images, self.uv_maps, self.extrinsics, self.intrinsics,\
        self.intrinsic_image_sizes, self.depth_images, self.size, self.scene_dict = self.parse_scenes()

    @abstractmethod
    def get_scenes(self):
        """
        Return names to all scenes for the dataset.
        """
        pass

    @abstractmethod
    def get_colors(self, scene_path):
        """
        Return absolute paths to all colors images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_depth(self, scene_path):
        """
        Return absolute paths to all depth images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_uvs(self, scene_path):
        """
        Return absolute paths to all uvmap images for the scene (sorted!)
        """
        pass

    def get_intrinsics(self, scene_path):
        """
        Return 3x3 numpy array as intrinsic K matrix for the scene and (W,H) image dimensions if available
        """
        return np.identity(4, dtype=np.float32), (0, 0)

    def parse_scenes(self):
        rgb_images = []
        depth_images = []
        uv_maps = []
        extrinsics_matrices = []
        intrinsic_matrices = []
        intrinsic_image_sizes = []
        scene_dict = {}

        scenes = self.get_scenes()
        if self.verbose:
            print("Collecting images...")
            scenes = tqdm(scenes)

        for scene in scenes:
            scene_path = join(self.root_path, scene)
            if os.path.isdir(scene_path):
                scene_dict[scene] = {
                    "path": scene_path,
                    "items": 0,
                }

                colors = self.get_colors(scene_path)
                depth = self.get_depth(scene_path)
                uvs = self.get_uvs(scene_path)

                extrinsics = self.get_extrinsics(scene_path)
                intrinsics, image_size = self.get_intrinsics(scene_path)
                intrinsics = [intrinsics for i in range(len(colors))]
                image_size = [image_size for i in range(len(colors))]

                if len(colors) > 0 and len(colors) == len(extrinsics) and\
                   len(extrinsics) == len(depth) and len(depth) == len(uvs):
                    rgb_images.extend(colors)
                    depth_images.extend(depth)
                    uv_maps.extend(uvs)

                    extrinsics_matrices.extend(extrinsics)
                    intrinsic_matrices.extend(intrinsics)
                    intrinsic_image_sizes.extend(image_size)
                    scene_dict[scene]["items"] = len(colors)
                    scene_dict[scene]["color"] = colors
                    scene_dict[scene]["depth"] = depth
                    scene_dict[scene]["extrinsics"] = extrinsics
                    scene_dict[scene]["intrinsics"] = intrinsics
                    scene_dict[scene]["image_size"] = image_size
                    scene_dict[scene]["uv_map"] = uvs
                elif self.verbose:
                    print(
                        f"Scene {scene_path} rendered incomplete --> is skipped. colors: {len(colors)}, uvs: {len(uvs)}, extr: {len(extrinsics)}, depth: {len(depth)}")

        return rgb_images, uv_maps, extrinsics_matrices, intrinsic_matrices,\
               intrinsic_image_sizes, depth_images, len(rgb_images), scene_dict

    @abstractmethod
    def load_extrinsics(self, idx):
        """
        load the extrinsics item from self.extrinsics

        :param idx: the item to load

        :return: the extrinsics as numpy array
        """
        pass

    def load_intrinsics(self, idx):
        """
        load the intrinsics item from self.intrinsics

        :param idx: the item to load

        :return: the intrinsics as numpy array
        """
        return self.intrinsics[idx]

    @abstractmethod
    def load_uvmap(self, idx, pyramid_idx=0):
        """
        load the uvmap item from self.uv_maps

        :param idx: the item to load

        :return: the uvmap as PIL image or numpy array
        """
        pass

    @abstractmethod
    def calculate_mask(self, uvmap, depth=None):
        """
        calculate the uvmap mask item from uvmap (valid values == 1)

        :param idx: the uvmap from which to calculate the mask

        :return: the mask as PIL image
        """
        pass

    def prepare_getitem(self, idx):
        """
        Implementations can prepare anything necessary for loading this idx, i.e. load a .hdf5 file
        :param idx:
        :return:
        """
        pass

    def finalize_getitem(self, idx):
        """
        Implementations can finalize anything necessary after loading this idx, i.e. close a .hdf5 file
        :param idx:
        :return:
        """
        pass

    def load_rgb(self, idx):
        return Image.open(self.rgb_images[idx])

    def load_depth(self, idx):
        return Image.open(self.depth_images[idx])

    def modify_intrinsics_matrix(self, intrinsics, intrinsics_image_size, rgb_image_size):
        if intrinsics_image_size != rgb_image_size:
            intrinsics = np.array(intrinsics)
            intrinsics[0, 0] = (intrinsics[0, 0] / intrinsics_image_size[0]) * rgb_image_size[0]
            intrinsics[1, 1] = (intrinsics[1, 1] / intrinsics_image_size[1]) * rgb_image_size[1]
            intrinsics[0, 2] = (intrinsics[0, 2] / intrinsics_image_size[0]) * rgb_image_size[0]
            intrinsics[1, 2] = (intrinsics[1, 2] / intrinsics_image_size[1]) * rgb_image_size[1]

        return intrinsics

    def __len__(self):
        return self.size

    def __getitem__(self, item, only_cam=False):
        if item not in self.cache or only_cam:
            self.prepare_getitem(item)

            extrinsics = self.load_extrinsics(item)
            extrinsics = torch.from_numpy(extrinsics)
            intrinsics = self.load_intrinsics(item)
            intrinsics = torch.from_numpy(intrinsics)

            rgb = self.load_rgb(item)
            depth = self.load_depth(item)
            uv = self.load_uvmap(item)
            mask = self.calculate_mask(uv, depth)

            if not self.resize:
                # only resize it if we do not have to resize everything afterwards anyways
                # resize the rgb, label, instance images to the size of uv to be consistent
                uv_size = np.asarray(uv).shape[:2]
                rgb = torchvision.transforms.Resize(uv_size, interpolation=Image.BICUBIC)(rgb)

            if self.resize:
                if isinstance(self.resize_size, int):
                    w, h = rgb.size
                    h_new = self.resize_size
                    w_new = round(w * h_new / h)
                    resize_size = (w_new, h_new)
                else:
                    resize_size = self.resize_size

                rgb = rgb.resize(resize_size)
                mask = mask.resize(resize_size, Image.NEAREST)

            # fix intrinsics to resized item
            #intrinsics = self.modify_intrinsics_matrix(intrinsics, self.intrinsic_image_sizes[item], rgb.size)
            #intrinsics = torch.from_numpy(intrinsics)

            if self.transform_rgb:
                rgb = self.transform_rgb(rgb)

            mask = torchvision.transforms.ToTensor()(mask)
            mask = mask > 0
            mask = mask.squeeze()  # Final shape: H x W

            result = (rgb, extrinsics, intrinsics, mask)

            if self.use_cache:
                self.cache[item] = result

            self.finalize_getitem(item)
            return result
        else:
            return self.cache[item]
