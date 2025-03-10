import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_wdepth_model import SR_wdepth_Model


@MODEL_REGISTRY.register()
class SwinIR_wdepth_Model(SR_wdepth_Model):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        depth = F.pad(self.depth, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img,depth)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img,depth)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        #判断self.output是否存在nan
        if torch.isnan(self.output).any():
            print('debug nan in output')
            print('self output min and max',torch.min(self.output),torch.max(self.output))