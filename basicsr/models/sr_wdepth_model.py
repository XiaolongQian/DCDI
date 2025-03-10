import torch
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from tqdm import tqdm
import numpy as np
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from os import path as osp
@MODEL_REGISTRY.register()
class SR_wdepth_Model(SRModel):

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            # print('debug sr model feed_data gt shape',self.gt.shape)
        if 'depth' in data:
            self.depth = data['depth'].to(self.device) #归一化到0-1
            # self.depth = torch.clamp(self.depth, min=0.2, max=10)
            # print('debug sr model feed_data depth shape',self.depth.shape)
        if 'gt_depth' in data:
            self.gt_depth = data['gt_depth'].to(self.device) #未归一化
            # self.gt_depth = torch.clamp(self.gt_depth, min=0.2, max=10)
            # print('debug sr model feed_data gt_depth shape',self.gt_depth.shape)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq,self.depth)
        #判断self.lq是否存在nan
        if torch.isnan(self.lq).any():
            print('debug nan in lq')
            print('self lq min and max',torch.min(self.lq),torch.max(self.lq))
        #判断self.depth是否存在nan
        if torch.isnan(self.depth).any():
            print('debug nan in depth')
            print('self depth min and max',torch.min(self.depth),torch.max(self.depth))
        #判断self.output是否存在nan
        if torch.isnan(self.output).any():
            print('debug nan in output')
            print('self output min and max',torch.min(self.output),torch.max(self.output))
        # print('debug depth weight matrix',self.opt['train'].get('depth_weight_matrix',False))

        # if self.opt['train'].get('depth_weight_matrix',False):
        #     # print('debug using depth_weight_matrix')


        #     # 计算 depth_weight_matrix
        #     base_weight = 2  # 在 gt_depth 为0.7时的目标值，保证在3以下
        #     beta = 0.5         # 控制变化速率的参数
        #     gamma = 2        # 中心点

        #     # 初始化 depth_weight_matrix
        #     self.depth_weight_matrix = torch.zeros_like(self.gt_depth)

        #     # 对非零深度值计算权重
        #     mask = self.gt_depth != 0

        #     # 计算 depth_weight_matrix
        #     # self.depth_weight_matrix[mask] = 1 + (base_weight - 1) * torch.exp(-beta * (self.gt_depth[mask] - gamma)**2)
        #     self.depth_weight_matrix[mask] = 1 + (base_weight - 1) * torch.exp(-beta * (self.gt_depth[mask] - gamma)**1)
        #     # print('debug min max depth_weight_matrix',torch.min(self.depth_weight_matrix),torch.max(self.depth_weight_matrix))
        #     self.gt = self.gt * self.depth_weight_matrix
        #     self.output = self.output * self.depth_weight_matrix

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq,self.depth)
                #判断self.output是否存在nan
                if torch.isnan(self.output).any():
                    print('debug nan in output')
                    print('self output min and max',torch.min(self.output),torch.max(self.output))

        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq,self.depth)
            self.net_g.train()

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
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            lq_img = tensor2img([visuals['lq']])

            depth = visuals['depth'].squeeze(0).squeeze(0).float().detach().cpu().numpy()
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output

            del self.depth
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    #png
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_img_lq_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                                             f'{img_name}_lq.png')

                    save_img_gt_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                                             f'{img_name}_gt.png')

                    save_depth_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                                             f'{img_name}_pred_depth.png')
                    # #jpg
                    # save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                    #                          f'{img_name}_{current_iter}.jpg')
                    # save_img_lq_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                    #                          f'{img_name}_lq.jpg')
                    # save_img_gt_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                    #                          f'{img_name}_gt.jpg')
                    # save_depth_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                    #                          f'{img_name}_pred_depth.jpg')

                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        # save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                        #                          f'{img_name}_{self.opt["name"]}.png')
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                                                 f'{img_name}_{self.opt["name"]}.png')

                        save_img_lq_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                                             f'{img_name}_lq.png')

                        save_img_gt_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                                             f'{img_name}_gt.png')
                        save_depth_path = osp.join(self.opt['path']['visualization'], dataset_name,img_name,
                                             f'{img_name}_depth.png')
                depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
                depth = (depth * 255).astype(np.uint8)
                imwrite(depth, save_depth_path)
                imwrite(sr_img, save_img_path)
                imwrite(lq_img, save_img_lq_path)
                imwrite(gt_img, save_img_gt_path)

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

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['depth'] = self.depth.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict