import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.models.sr_model import SRModel

@MODEL_REGISTRY.register()
class PSF_Restoration_Model(SRModel):
    """Base SR model for single image super-resolution."""

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # index_H,index_W,index_D = val_data['key'][0].split(' ')
            index_H,index_W,index_D,index_N = val_data['key'][0].split(' ')
            lens_name = 'lens'+index_N
            img_name = 'index_H'+index_H+'index_W'+index_W+'index_D'+index_D
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # print('debug model sr img',visuals['result'].shape)
            # print('debug model gt img',visuals['gt'].shape)
            # import numpy as np
            # print('debug model gt img',torch.max(visuals['gt']),torch.min(visuals['gt']))

            sr_img = tensor2img([visuals['result']])
            # print('debug model sr_img',sr_img.shape)
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                # print('debug model gt_img',gt_img.shape)
                # import numpy as np
                # print('debug model gt_img',np.max(gt_img),np.min(gt_img))
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    # save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                    #                          f'{img_name}_{current_iter}.png')
                    save_img_path = osp.join(self.opt['path']['visualization'],lens_name ,img_name,
                                             f'{img_name}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'], lens_name,img_name,
                                             f'{img_name}_gt.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        # save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                        #                          f'{img_name}_{self.opt["name"]}.png')
                        # save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name,f'{img_name}_gt.png')
                        save_img_path = osp.join(self.opt['path']['visualization'],lens_name ,img_name,
                                                f'{img_name}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'], lens_name,img_name,
                                                f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
