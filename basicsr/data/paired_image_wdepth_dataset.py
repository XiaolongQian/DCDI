from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_depth,augment_depth,paired_random_crop_depth_gtdepth, augment_depth_gtdepth,normalize_depth_image,limit_tensor_values,normalize_depth_imagev2,paired_random_crop_depthv2
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import os
import numpy as np
import torch
@DATASET_REGISTRY.register()
class PairedImage_wdepth_Dataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImage_wdepth_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.depth_map = np.load(opt['depth_map'])

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        #读取是否归一化深度图，默认设置为False
        self.norm_depth_map = opt['norm_depth_map'] if 'norm_depth_map' in opt else False
        self.division_depth_map = opt['division_depth_map'] if 'division_depth_map' in opt else False
        # print("if norm_depth_map: ",self.norm_depth_map)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        #get_depth_map
        img_name_with_extension = os.path.basename(gt_path)
        img_name = os.path.splitext(img_name_with_extension)[0]
        depth_map = self.depth_map[int(img_name)]
        depth_map = np.expand_dims(depth_map, axis=2)

        #img_gt 存在0值的地方 生成mask
        mask = (img_gt == 0).all(axis=2)
        #对depth进行mask 不改变shape
        depth_map[mask] = 0
        # print('mask:',mask.shape)
        # print('depth_map:',depth_map.shape)
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, depth_map = paired_random_crop_depth(img_gt, img_lq,depth_map, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq,depth_map = augment_depth([img_gt, img_lq, depth_map], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        depth_map = torch.from_numpy(depth_map.copy().transpose(2, 0, 1)).float()

        #depth_map 存在nan就暂停
        assert not torch.isnan(depth_map).any(), 'primary depth_map has nan value!'

        #lens1
        depth_map = limit_tensor_values(depth_map,min_value=0.1,max_value=10)

        #depth_map 存在nan就暂停
        assert not torch.isnan(depth_map).any(), 'limit depth_map has nan value!'

        #normalize depth_map [0,1]
        #如果norm_depth_map和division_depth_map相等，就报错 assert
        if self.norm_depth_map == self.division_depth_map:
            assert 'norm_depth_map and division_depth_map can not be equal!'

        if self.norm_depth_map:

            depth_map = normalize_depth_image(depth_map)
            #depth_map 存在nan就暂停
            if torch.isnan(depth_map).any():
                print('index',index,'lq_path',lq_path,'gt_path',gt_path,'depth_map',img_name_with_extension)
            assert not torch.isnan(depth_map).any(), 'norm depth_map has nan value!'

        if self.division_depth_map:
            depth_map = depth_map / 10.0

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path,'depth':depth_map}

    def __len__(self):
        return len(self.paths)
