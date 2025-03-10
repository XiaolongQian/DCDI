import cv2
import random
import torch
import numpy as np

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def paired_random_crop_psf(img_gt, img_lq, psf,patch_size, scale, gt_path=None):
    """
    The function takes as input two images and a tensor, and returns
    cropped versions of these images  and the tensor. The cropping
    is performed in such a way that the corresponding areas in img_gt,
    img_lq and psf are cropped.

    Args:
        img_gt (numpy array): GT image.
        img_lq (numpy array): LQ image.
        psf (torch tensor) : A tensor.
        patch_size (int): The size of the patch to crop from the images.

    Returns:
        img_gt (numpy array): Cropped GT image.
        img_lq (numpy array): Cropped LQ image.
        psf (torch tensor) : Cropped tensor.
    """

    # the height and the width of the images
    h, w = img_lq.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError('Both dimensions of the images should be greater than the patch size.')

    # randomly choose top and left coordinates for patch
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    # crop patch
    img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]

    # crop psf in the corresponding area
    psf = psf[:, top:top + patch_size, left:left + patch_size]

    return img_gt, img_lq, psf

def paired_random_crop_depth_psf(img_gt, img_lq, depth_map,psf,patch_size, scale, gt_path=None):
    """
    The function takes as input two images and a tensor, and returns
    cropped versions of these images  and the tensor. The cropping
    is performed in such a way that the corresponding areas in img_gt,
    img_lq and psf are cropped.

    Args:
        img_gt (numpy array): GT image.
        img_lq (numpy array): LQ image.
        psf (torch tensor) : A tensor.
        patch_size (int): The size of the patch to crop from the images.

    Returns:
        img_gt (numpy array): Cropped GT image.
        img_lq (numpy array): Cropped LQ image.
        psf (torch tensor) : Cropped tensor.
    """

    # the height and the width of the images
    h, w = img_lq.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError('Both dimensions of the images should be greater than the patch size.')

    # randomly choose top and left coordinates for patch
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    # crop patch
    img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]
    # crop psf in the corresponding area
    psf = psf[:, top:top + patch_size, left:left + patch_size]

    return img_gt, img_lq, depth_map,psf

def paired_random_crop_depth_psf_fwhm(img_gt, img_lq, depth_map,psf,fwhm,patch_size, scale, gt_path=None):
    """
    The function takes as input two images and a tensor, and returns
    cropped versions of these images  and the tensor. The cropping
    is performed in such a way that the corresponding areas in img_gt,
    img_lq and psf are cropped.

    Args:
        img_gt (numpy array): GT image.
        img_lq (numpy array): LQ image.
        psf (torch tensor) : A tensor.
        patch_size (int): The size of the patch to crop from the images.

    Returns:
        img_gt (numpy array): Cropped GT image.
        img_lq (numpy array): Cropped LQ image.
        psf (torch tensor) : Cropped tensor.
    """

    # the height and the width of the images
    h, w = img_lq.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError('Both dimensions of the images should be greater than the patch size.')

    # randomly choose top and left coordinates for patch
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    # crop patch
    img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]
    fwhm = fwhm[top:top + patch_size, left:left + patch_size,:]
    # crop psf in the corresponding area
    psf = psf[:, top:top + patch_size, left:left + patch_size]

    return img_gt, img_lq, depth_map,psf,fwhm

def paired_random_crop_depth(img_gt, img_lq, depth_map,patch_size, scale, gt_path=None):
    """
    The function takes as input two images and a tensor, and returns
    cropped versions of these images  and the tensor. The cropping
    is performed in such a way that the corresponding areas in img_gt,
    img_lq and psf are cropped.

    Args:
        img_gt (numpy array): GT image.
        img_lq (numpy array): LQ image.
        psf (torch tensor) : A tensor.
        patch_size (int): The size of the patch to crop from the images.

    Returns:
        img_gt (numpy array): Cropped GT image.
        img_lq (numpy array): Cropped LQ image.
        psf (torch tensor) : Cropped tensor.
    """

    # the height and the width of the images
    h, w = img_lq.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError('Both dimensions of the images should be greater than the patch size.')

    # primary crop patch
    # randomly choose top and left coordinates for patch
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    # crop patch
    img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]

    # #lens1
    # valid_crop = False
    # while not valid_crop:
    #     # randomly choose top and left coordinates for patch
    #     top = random.randint(0, h - patch_size)
    #     left = random.randint(0, w - patch_size)

    #     # temporarily crop depth_map to check conditions
    #     temp_depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]

    #     # check if more than 50% of the values in depth_map are 0
    #     # print('zero_percent',np.sum(temp_depth_map == 0) / (patch_size * patch_size))
    #     # if np.sum(temp_depth_map == 0) / (patch_size * patch_size) > 0.05:
    #     #     print('retrying crop')
    #     if np.sum(temp_depth_map == 0) / (patch_size * patch_size) <= 0.05:

    #         valid_crop = True
    #         img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    #         img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    #         depth_map = temp_depth_map
    #     # else, loop will retry cropping

    # #lens3
    # valid_crop = False
    # while not valid_crop:
    #     # randomly choose top and left coordinates for patch
    #     top = random.randint(0, h - patch_size)
    #     left = random.randint(0, w - patch_size)

    #     # temporarily crop depth_map to check conditions
    #     temp_depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]

    #     # check if more than 50% of the values in depth_map are 0
    #     # print('zero_percent',np.sum(temp_depth_map == 0) / (patch_size * patch_size))
    #     if np.sum(temp_depth_map == 0) / (patch_size * patch_size) > 0.2:
    #         print('retrying crop')
    #     if np.sum(temp_depth_map == 0) / (patch_size * patch_size) <= 0.2:

    #         valid_crop = True
    #         img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    #         img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    #         depth_map = temp_depth_map
    #     # else, loop will retry cropping


    return img_gt, img_lq, depth_map

def paired_random_crop_depthv2(img_gt, img_lq, depth_map,patch_size, scale, gt_path=None):
    """
    The function takes as input two images and a tensor, and returns
    cropped versions of these images  and the tensor. The cropping
    is performed in such a way that the corresponding areas in img_gt,
    img_lq and psf are cropped.

    Args:
        img_gt (numpy array): GT image.
        img_lq (numpy array): LQ image.
        psf (torch tensor) : A tensor.
        patch_size (int): The size of the patch to crop from the images.

    Returns:
        img_gt (numpy array): Cropped GT image.
        img_lq (numpy array): Cropped LQ image.
        psf (torch tensor) : Cropped tensor.
    """

    # the height and the width of the images
    h, w = img_lq.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError('Both dimensions of the images should be greater than the patch size.')

    # primary crop patch
    # randomly choose top and left coordinates for patch
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    # crop patch
    img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]

    # #lens1
    # valid_crop = False
    # while not valid_crop:
    #     # randomly choose top and left coordinates for patch
    #     top = random.randint(0, h - patch_size)
    #     left = random.randint(0, w - patch_size)

    #     # temporarily crop depth_map to check conditions
    #     temp_depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]

    #     # check if more than 50% of the values in depth_map are 0
    #     # print('zero_percent',np.sum(temp_depth_map == 0) / (patch_size * patch_size))
    #     # if np.sum(temp_depth_map == 0) / (patch_size * patch_size) > 0.05:
    #     #     print('retrying crop')
    #     if np.sum(temp_depth_map == 0) / (patch_size * patch_size) <= 0.05:

    #         valid_crop = True
    #         img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    #         img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    #         depth_map = temp_depth_map
    #     # else, loop will retry cropping

    # #lens3
    # valid_crop = False
    # while not valid_crop:
    #     # randomly choose top and left coordinates for patch
    #     top = random.randint(0, h - patch_size)
    #     left = random.randint(0, w - patch_size)

    #     # temporarily crop depth_map to check conditions
    #     temp_depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]

    #     # check if more than 50% of the values in depth_map are 0
    #     # print('zero_percent',np.sum(temp_depth_map == 0) / (patch_size * patch_size))
    #     if np.sum(temp_depth_map == 0) / (patch_size * patch_size) > 0.2:
    #         print('retrying crop')
    #     if np.sum(temp_depth_map == 0) / (patch_size * patch_size) <= 0.2:

    #         valid_crop = True
    #         img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    #         img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    #         depth_map = temp_depth_map
    #     # else, loop will retry cropping


    return img_gt, img_lq, depth_map

def paired_random_crop_depth_gtdepth(img_gt, img_lq, depth_map,gt_depth_map,patch_size, scale, gt_path=None):
    """
    The function takes as input two images and a tensor, and returns
    cropped versions of these images  and the tensor. The cropping
    is performed in such a way that the corresponding areas in img_gt,
    img_lq and psf are cropped.

    Args:
        img_gt (numpy array): GT image.
        img_lq (numpy array): LQ image.
        psf (torch tensor) : A tensor.
        patch_size (int): The size of the patch to crop from the images.

    Returns:
        img_gt (numpy array): Cropped GT image.
        img_lq (numpy array): Cropped LQ image.
        psf (torch tensor) : Cropped tensor.
    """

    # the height and the width of the images
    h, w = img_lq.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError('Both dimensions of the images should be greater than the patch size.')

    # randomly choose top and left coordinates for patch
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    # crop patch
    img_gt = img_gt[top:top + patch_size, left:left + patch_size,:]
    img_lq = img_lq[top:top + patch_size, left:left + patch_size,:]
    depth_map = depth_map[top:top + patch_size, left:left + patch_size,:]
    gt_depth_map = gt_depth_map[top:top + patch_size, left:left + patch_size,:]
    # crop psf in the corresponding area
    # psf = psf[:, top:top + patch_size, left:left + patch_size]

    return img_gt, img_lq, depth_map, gt_depth_map

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def augment_psf(imgs, psf, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        psf (torch.Tensor): PSF tensor to be augmented.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and PSF tensor. If returned
            results only have one element, just return ndarray.

    """

    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = np.flip(img, axis=1)
        if vflip:  # vertical
            img = np.flip(img, axis=0)
        if rot90:
            img = np.rot90(img)
        return img

    def _augment_psf(psf):
        if hflip:  # horizontal
            psf = psf.flip(-1)
        if vflip:  # vertical
            psf = psf.flip(-2)
        if rot90:
            psf = psf.transpose(-1, -2)
        return psf

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    psf = _augment_psf(psf)

    if return_status:
        return imgs, psf, (hflip, vflip, rot90)
    else:
        return imgs[0],imgs[1], psf

def augment_depth_psf(imgs, psf, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        psf (torch.Tensor): PSF tensor to be augmented.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and PSF tensor. If returned
            results only have one element, just return ndarray.

    """

    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = np.flip(img, axis=1)
        if vflip:  # vertical
            img = np.flip(img, axis=0)
        if rot90:
            img = np.rot90(img)
        return img

    def _augment_psf(psf):
        if hflip:  # horizontal
            psf = psf.flip(-1)
        if vflip:  # vertical
            psf = psf.flip(-2)
        if rot90:
            psf = psf.transpose(-1, -2)
        return psf

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    psf = _augment_psf(psf)

    if return_status:
        return imgs, psf, (hflip, vflip, rot90)
    else:
        return imgs[0],imgs[1],imgs[2], psf

def augment_depth_psf_fwhm(imgs, psf, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        psf (torch.Tensor): PSF tensor to be augmented.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and PSF tensor. If returned
            results only have one element, just return ndarray.

    """

    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = np.flip(img, axis=1)
        if vflip:  # vertical
            img = np.flip(img, axis=0)
        if rot90:
            img = np.rot90(img)
        return img

    def _augment_psf(psf):
        if hflip:  # horizontal
            psf = psf.flip(-1)
        if vflip:  # vertical
            psf = psf.flip(-2)
        if rot90:
            psf = psf.transpose(-1, -2)
        return psf

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    psf = _augment_psf(psf)

    if return_status:
        return imgs, psf, (hflip, vflip, rot90)
    else:
        return imgs[0],imgs[1],imgs[2],imgs[3], psf

def augment_depth(imgs, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        psf (torch.Tensor): PSF tensor to be augmented.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and PSF tensor. If returned
            results only have one element, just return ndarray.

    """

    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = np.flip(img, axis=1)
        if vflip:  # vertical
            img = np.flip(img, axis=0)
        if rot90:
            img = np.rot90(img)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]


    if return_status:
        return imgs, (hflip, vflip, rot90)
    else:
        return imgs[0],imgs[1],imgs[2]

def augment_depth_gtdepth(imgs, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        psf (torch.Tensor): PSF tensor to be augmented.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and PSF tensor. If returned
            results only have one element, just return ndarray.

    """

    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = np.flip(img, axis=1)
        if vflip:  # vertical
            img = np.flip(img, axis=0)
        if rot90:
            img = np.rot90(img)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]


    if return_status:
        return imgs, (hflip, vflip, rot90)
    else:
        return imgs[0],imgs[1],imgs[2],imgs[3]

def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

def normalize_depth_image(depth_image):
    depth_image = depth_image.float()

    # 创建掩码，用于标记非缺失值的部分
    mask = depth_image != 0

    # 计算有效值的最小值和最大值
    valid_min = torch.min(depth_image[mask])
    valid_max = torch.max(depth_image[mask])

    # 归一化深度值
    depth_image_normalized = (depth_image - valid_min) / (valid_max - valid_min)

    # 对缺失值（原为0的地方）保持为0
    depth_image_normalized[~mask] = 0

    return depth_image_normalized

def normalize_depth_imagev2(depth_image):
    depth_image = depth_image.float()

    # 创建掩码，用于标记非缺失值的部分
    mask = depth_image != 0

    # 计算有效值的最小值和最大值
    valid_min = torch.min(depth_image[mask])
    valid_max = torch.max(depth_image[mask])

    # 归一化深度值
    # depth_image_normalized = (depth_image - valid_min) / (valid_max - valid_min)
    depth_image_normalized = depth_image / valid_max

    # 对缺失值（原为0的地方）保持为0
    depth_image_normalized[~mask] = 0

    return depth_image_normalized

# def limit_tensor_values(tensor,min_value=0.7,max_value=10):
#     tensor = tensor.float()
#     # 将缺失值（0）标记为NaN
#     tensor[tensor == 0] = float('nan')

#     # 将有值的地方限制在0.7到10之间
#     tensor = torch.clamp(tensor, min=min_value, max=max_value)

#     # 将NaN值恢复为0
#     tensor[torch.isnan(tensor)] = 0

#     return tensor

import torch

def limit_tensor_values(tensor, min_value=0.7, max_value=10):
    # 复制一个tensor来操作，以保留原始tensor不变
    result_tensor = tensor.clone()
    # 找出所有非0的位置
    non_zero_indices = result_tensor != 0
    # 仅在非0的位置应用clamp操作，将值限制在min_value和max_value之间
    result_tensor[non_zero_indices] = torch.clamp(result_tensor[non_zero_indices], min=min_value, max=max_value)
    return result_tensor
#主函数
if __name__ == '__main__':
    #测试limit_tensor_values函数
    tensor = torch.tensor([0.1, 0.3, 0.7, 0, 0, 0.7, 0.7, 0])
    print(tensor)
    tensor = limit_tensor_values(tensor)
    print(tensor)
    #测试normalize_depth_image函数
    tensor = normalize_depth_image(tensor)
    print(tensor)

