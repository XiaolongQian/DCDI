import numpy as np
import torch
from pathlib import Path
from torch.utils import data as data
import h5py
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import FileClient


@DATASET_REGISTRY.register()
class Paired_XYZ2PSF_Dataset(data.Dataset):
    """Paired XYZF dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths.
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
        super(Paired_XYZ2PSF_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_root = Path(opt['dataroot_gt'])

        self.patchlength = opt['patchlength']
        self.img_size = opt['img_size']
        self.H_hum = self.img_size[0] // self.patchlength
        self.W_hum = self.img_size[1] // self.patchlength
        self.max_Depth = opt['max_depth']
        # D_hum = opt['depth']
        self.psf_size = opt['psf_size']
        self.keys = []
        # D_values_1 = np.arange(1.4, 3, 0.2)
        # D_values_2 = np.arange(3, 21, 1)
        # D_values = np.concatenate((D_values_1, D_values_2))
        D_values = np.arange(0.2, 10.1, 0.1)
        print('debug ',D_values)
        # self.keys = [f"{H}/{W}/{D}" for H in range(self.H_hum) for W in range(self.W_hum) for D in range(2, 21)]  # 80x120x19 H/W/D

        if self.opt['phase'] == 'train':
            self.keys = []
            # self.keys = [f"{H:0>3} {W:0>3} {D:0>3}" for H in range(self.H_hum) for W in range(self.W_hum) for D in range(2, self.Depth + 1)]
            self.keys = [f"{H:0>3} {W:0>3} {D:.1f}" for H in range(self.H_hum) for W in range(self.W_hum) for D in D_values]
            # self.keys = [f"{H:0>3} {W:0>3} {D:0>3}" for H in range(2) for W in range(4) for D in range(2, 15)]
            # print(self.keys)
            print(len(self.keys))
            # remove some point used in validation
            to_delete = set()

            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    # 移除行尾的换行符并将其添加到set中
                    to_delete.add(line.strip())
                # print(to_delete)

            # 过滤self.keys，只保留那些没有在meta_info_file中列出的元素
            self.keys = sorted([key for key in self.keys if key not in to_delete])
            # print(self.keys)
            print(len(self.keys))

        if self.opt['phase'] == 'val':
            self.keys = []
            # move some point used in validation
            to_add = set()

            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    # 移除行尾的换行符并将其添加到set中
                    to_add.add(line.strip())
                # print(to_add)

            # 过滤self.keys，只保留那些在meta_info_file中列出的元素
            self.keys = sorted([key for key in to_add])
            # print(self.keys)
            print(len(self.keys))


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        # print('debug key',key)

        index_H,index_W,index_D = key.split(' ')
        # print('debug index',index_H,index_W,index_D)
        # H = self.patchlength * int(index_H)
        # W = self.patchlength * int(index_W)
        # # print('debug index_D',index_D)
        # # print('debug float index_D',float(index_D))
        # # D = int((-500) * round(float(index_D),1))
        # D = float(index_D)
        # # print('debug D',D)
        # # fd = 28.3
        # # 对 H W D 进行归一化
        # norm_H = H / self.img_size[0]
        # norm_W = W / self.img_size[1]
        # # norm_D = D / -5000
        # norm_D = D / -10000
        norm_H = float(index_H) / (self.H_hum - 1)
        norm_W = float(index_W) / (self.W_hum - 1)
        norm_D = float(index_D) / self.max_Depth
        # float32
        norm_H = np.float32(norm_H)
        norm_W = np.float32(norm_W)
        norm_D = np.float32(norm_D)
        # input_feature = [H,W,D,fd]
        input_feature = [norm_H,norm_W,norm_D]
        # print('debug input_feature',input_feature)
        # print('debug h w d',H,W,D)
        #找到对应深度的PSF MAP
        gt_psf_path = self.gt_root / 'depth_aware_psfmap_standard.h5'
        #open hdf5 file

        # f = h5py.File(gt_psf_path, 'r')
        # group = f['lens1']
        # dataset_name = 'depth_' + str(D)
        # all_gt_psf_map = group[dataset_name][:]
        # # print('debug psf map',all_gt_psf_map.shape)
        # all_gt_psf_map = all_gt_psf_map.reshape(self.H_hum * self.W_hum,3,self.psf_size,self.psf_size)
        # # print('debug psf map',all_gt_psf_map.shape)
        # # print('debug ',index_W,index_H,self.H_hum,self.W_hum)
        # gt_psf_map = all_gt_psf_map[int(index_H) * self.W_hum + int(index_W)] # 3x41x41
        D = int(float(index_D) * 1000)
        f = h5py.File(gt_psf_path, 'r')
        group = f['lens1']
        dataset_name = 'depth_-' + str(D)
        gt_psf_map_index = int(index_H) * self.W_hum * 3+ int(index_W) * 3
        gt_psf_map = group[dataset_name][gt_psf_map_index:gt_psf_map_index+3]
        # print('debug psf map',gt_psf_map.shape)
        gt_psf_map = gt_psf_map.reshape(3,self.psf_size,self.psf_size)
        # print('debug psf map',gt_psf_map.shape)

        # # 判断是否相等
        # print('debug',np.all(gt_psf_map == gt_psf_map1))
        # channel_sums = np.sum(gt_psf_map, axis=(1, 2)) #psf 每个通道的和为1
        # print('debug channel sums',channel_sums)
        # 把psf小于0的值置0
        gt_psf_map = np.maximum(gt_psf_map, 0)
        # psf 每个通道归一化到 [0,1]
        gt_psf_map = gt_psf_map / np.max(gt_psf_map, axis=(1, 2), keepdims=True)

        # print('debug psf map',gt_psf_map.shape)
        # numpy -> tensor
        gt_psf_map = torch.tensor(gt_psf_map, dtype=torch.float32)
        input_feature = torch.tensor(input_feature, dtype=torch.float32)

        # print('debug gt_psf_map',gt_psf_map.shape)
        # print('debug',np.max(gt_psf_map.numpy()),np.min(gt_psf_map.numpy()))
        # print('debug input_feature',input_feature.shape)

        # from torchvision import utils
        # utils.save_image(gt_psf_map.unsqueeze(0), 'debug_psf_map.png',normalize=True)

        return {'lq': input_feature, 'gt': gt_psf_map,'key':key}

    def __len__(self):
        # print('debug len',len(self.keys))
        return len(self.keys)

@DATASET_REGISTRY.register()
class Paired_XYZN2PSF_Dataset(data.Dataset):
    """Paired XYZF dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths.
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
        super(Paired_XYZN2PSF_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_root = Path(opt['dataroot_gt'])

        self.patchlength = opt['patchlength']
        self.img_size = opt['img_size']
        self.H_hum = self.img_size[0] // self.patchlength
        self.W_hum = self.img_size[1] // self.patchlength
        self.max_Depth = opt['max_depth']
        # D_hum = opt['depth']
        self.psf_size = opt['psf_size']
        self.keys = []
        # D_values_1 = np.arange(1.4, 3, 0.2)
        # D_values_2 = np.arange(3, 21, 1)
        # D_values = np.concatenate((D_values_1, D_values_2))
        D_values = np.arange(0.7, 10.1, 0.1)
        print('debug ',D_values)
        self.N_lens = opt['N_lens']
        # self.keys = [f"{H}/{W}/{D}" for H in range(self.H_hum) for W in range(self.W_hum) for D in range(2, 21)]  # 80x120x19 H/W/D

        if self.opt['phase'] == 'train':
            self.keys = []
            # self.keys = [f"{H:0>3} {W:0>3} {D:0>3}" for H in range(self.H_hum) for W in range(self.W_hum) for D in range(2, self.Depth + 1)]
            # self.keys = [f"{H:0>3} {W:0>3} {D:.1f} {N:0>3}" for H in range(self.H_hum) for W in range(self.W_hum) for D in D_values for N in range(N_lens)]
            self.keys = [f"{H:0>3} {W:0>3} {D:.1f} {N:0>3}" for H in range(self.H_hum) for W in range(self.W_hum) for D in D_values for N in range(self.N_lens)]
            # self.keys = [f"{H:0>3} {W:0>3} {D:0>3}" for H in range(2) for W in range(4) for D in range(2, 15)]
            # print(self.keys)
            print(len(self.keys))
            # remove some point used in validation
            to_delete = set()

            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    # 移除行尾的换行符并将其添加到set中
                    to_delete.add(line.strip())
                # print(to_delete)

            # 过滤self.keys，只保留那些没有在meta_info_file中列出的元素
            self.keys = sorted([key for key in self.keys if key not in to_delete])
            # print(self.keys)
            print(len(self.keys))

        if self.opt['phase'] == 'val':
            self.keys = []
            # move some point used in validation
            to_add = set()

            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    # 移除行尾的换行符并将其添加到set中
                    to_add.add(line.strip())
                # print(to_add)

            # 过滤self.keys，只保留那些在meta_info_file中列出的元素
            self.keys = sorted([key for key in to_add])
            # print(self.keys)
            print(len(self.keys))

        if self.opt['phase'] == 'test':
            self.keys = []
            # move some point used in validation
            to_add = set()

            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    # 移除行尾的换行符并将其添加到set中
                    to_add.add(line.strip())
                # print(to_add)

            # 过滤self.keys，只保留那些在meta_info_file中列出的元素
            self.keys = sorted([key for key in to_add])
            # print(self.keys)
            print(len(self.keys))


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        # print('debug key',key)

        index_H,index_W,index_D,index_N = key.split(' ')
        # print('debug index',index_H,index_W,index_D)
        # H = self.patchlength * int(index_H)
        # W = self.patchlength * int(index_W)
        # # print('debug index_D',index_D)
        # # print('debug float index_D',float(index_D))
        # # D = int((-500) * round(float(index_D),1))
        # D = float(index_D)
        # # print('debug D',D)
        # # fd = 28.3
        # # 对 H W D 进行归一化
        # norm_H = H / self.img_size[0]
        # norm_W = W / self.img_size[1]
        # # norm_D = D / -5000
        # norm_D = D / -10000
        norm_H = float(index_H) / (self.H_hum - 1)
        norm_W = float(index_W) / (self.W_hum - 1)
        norm_D = float(index_D) / self.max_Depth
        norm_N = float(index_N) / (self.N_lens - 1)
        # float32
        norm_H = np.float32(norm_H)
        norm_W = np.float32(norm_W)
        norm_D = np.float32(norm_D)
        norm_N = np.float32(norm_N)
        # input_feature = [H,W,D,fd]
        input_feature = [norm_H,norm_W,norm_D,norm_N]
        # print('debug input_feature',input_feature)
        # print('debug h w d',H,W,D)
        #根据index_N选择对应的PSF MAP
        if index_N == '000':
            gt_psf_path = self.gt_root /'LENS1'/ 'depth_aware_psfmap_standard.h5'
            group_name = 'lens1'
        elif index_N == '001':
            gt_psf_path = self.gt_root /'LENS4'/ 'depth_aware_psfmap_standard.h5'
            group_name = 'lens4'
        elif index_N == '002':
            gt_psf_path = self.gt_root /'DoubleGauss'/ 'depth_aware_psfmap_double_gauss_standard.h5'
            group_name = 'lens5'
        elif index_N == '003':
            gt_psf_path = self.gt_root /'Piece6_fov32_F1.8'/ 'depth_aware_psfmap_piece6_fov32_F1.8_standard.h5'
            group_name = 'lens6'
        # #找到对应深度的PSF MAP
        # gt_psf_path = self.gt_root / 'depth_aware_psfmap_standard.h5'
        #open hdf5 file

        # f = h5py.File(gt_psf_path, 'r')
        # group = f['lens1']
        # dataset_name = 'depth_' + str(D)
        # all_gt_psf_map = group[dataset_name][:]
        # # print('debug psf map',all_gt_psf_map.shape)
        # all_gt_psf_map = all_gt_psf_map.reshape(self.H_hum * self.W_hum,3,self.psf_size,self.psf_size)
        # # print('debug psf map',all_gt_psf_map.shape)
        # # print('debug ',index_W,index_H,self.H_hum,self.W_hum)
        # gt_psf_map = all_gt_psf_map[int(index_H) * self.W_hum + int(index_W)] # 3x41x41
        D = int(float(index_D) * 1000)
        f = h5py.File(gt_psf_path, 'r')
        group = f[group_name]
        dataset_name = 'depth_-' + str(D)
        gt_psf_map_index = int(index_H) * self.W_hum * 3 + int(index_W) * 3
        gt_psf_map = group[dataset_name][gt_psf_map_index:gt_psf_map_index+3]
        # print('debug psf map',gt_psf_map.shape)
        gt_psf_map = gt_psf_map.reshape(3,self.psf_size,self.psf_size)
        # print('debug psf map',gt_psf_map.shape)

        # # 判断是否相等
        # print('debug',np.all(gt_psf_map == gt_psf_map1))
        # channel_sums = np.sum(gt_psf_map, axis=(1, 2)) #psf 每个通道的和为1
        # print('debug channel sums',channel_sums)
        # 把psf小于0的值置0
        gt_psf_map = np.maximum(gt_psf_map, 0)
        # psf 每个通道归一化到 [0,1]
        gt_psf_map = gt_psf_map / np.max(gt_psf_map, axis=(1, 2), keepdims=True)

        # print('debug psf map',gt_psf_map.shape)
        # numpy -> tensor
        gt_psf_map = torch.tensor(gt_psf_map, dtype=torch.float32)
        input_feature = torch.tensor(input_feature, dtype=torch.float32)

        # print('debug gt_psf_map',gt_psf_map.shape)
        # print('debug',np.max(gt_psf_map.numpy()),np.min(gt_psf_map.numpy()))
        # print('debug input_feature',input_feature.shape)

        # from torchvision import utils
        # utils.save_image(gt_psf_map.unsqueeze(0), 'debug_psf_map.png',normalize=True)

        return {'lq': input_feature, 'gt': gt_psf_map,'key':key}

    def __len__(self):
        # print('debug len',len(self.keys))
        return len(self.keys)