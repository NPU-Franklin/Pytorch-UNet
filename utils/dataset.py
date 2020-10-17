import logging
from glob import glob
from os import listdir
from os.path import splitext

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = 1000000000


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='_mask'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info('Creating dataset with {length} examples'.format(length=len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.jpg')
        img_file = glob(self.imgs_dir + idx + '.jpg')

        assert len(mask_file) == 1, \
            'Either no mask or multiple masks found for the ID {idx}: {mask_file}'.format(idx=idx, mask_file=mask_file)
        assert len(img_file) == 1, \
            'Either no image or multiple images found for the ID {idx}: {img_file}'.format(idx=idx, img_file=img_file)
        mask = Image.open(mask_file[0]).resize((5000, 5000), Image.ANTIALIAS)
        img = Image.open(img_file[0]).resize((5000, 5000), Image.ANTIALIAS)

        assert img.size == mask.size, \
            'Image and mask {idx} should be the same size, but are {img_size} and {mask_size}'.format(idx=idx,
                                                                                                      img_size=img.size,
                                                                                                      mask_size=mask.size)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
