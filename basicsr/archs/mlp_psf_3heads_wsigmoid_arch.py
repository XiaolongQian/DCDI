import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import Upsample, make_layer
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

@ARCH_REGISTRY.register()
class mlp_psf_3heads_w_sigmoid(nn.Module):
    """ All-linear layer. This network suits for low-k intensity/amplitude PSF function prediction.
    """
    def __init__(self, in_features, psf_channels=3,pred_psf_size=41,hidden_features=2048, hidden_layers=5):
        super(mlp_psf_3heads_w_sigmoid, self).__init__()

        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features//4, bias=True))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features//4, hidden_features, bias=True))
        self.net.append(nn.ReLU(inplace=True))
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features, bias=True))
            self.net.append(nn.ReLU(inplace=True))
        # self.net.append(nn.Linear(hidden_features, out_features, bias=True))
        # self.net.append(nn.Linear(hidden_features//2, hidden_features, bias=True))
        # self.net.append(nn.Linear(hidden_features, psf_channels*pred_psf_size*pred_psf_size, bias=True))
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

        self.head_r = nn.Sequential(
            nn.Linear(hidden_features, pred_psf_size*pred_psf_size, bias=True),
            nn.Sigmoid()
        )

        self.head_g = nn.Sequential(
            nn.Linear(hidden_features, pred_psf_size*pred_psf_size, bias=True),
            nn.Sigmoid()
        )

        self.head_b = nn.Sequential(
            nn.Linear(hidden_features, pred_psf_size*pred_psf_size, bias=True),
            nn.Sigmoid()
        )

        self.net.apply(initialize_weights)
        self.head_r.apply(initialize_weights)
        self.head_g.apply(initialize_weights)
        self.head_b.apply(initialize_weights)

        self.out_channels = psf_channels
        self.psf_size = pred_psf_size
    def forward(self, x):

        x = self.net(x)

        r_psf = self.head_r(x)

        g_psf = self.head_g(x)

        b_psf = self.head_b(x)

        output = torch.cat((r_psf,g_psf,b_psf),dim=1)

        output = output.reshape([-1,self.out_channels,self.psf_size,self.psf_size])
        # print("debug arch",torch.max(output),torch.min(output))
        # x = F.normalize(x, p=1, dim=-1)
        # print(x.shape)
        # x = x.reshape([-1,self.out_channels,self.psf_size,self.psf_size])
        # print(torch.max(x),torch.min(x))
        return output



if __name__ == '__main__':
    # net = mlp_psf_3heads_w_sigmoid(4)
    # x = torch.randn(8, 4)
    # out = net(x)
    # print(net)
    # print(out.shape)

    # net = mlp_psf_xyzn_3heads_w_sigmoid(4)
    # x = torch.randn(8, 4)
    # out = net(x)
    # print(net)
    # print(out.shape)


    ####inference
    import time
    import numpy as np
    import torch  # 假设您使用 PyTorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:',device)

    model = mlp_psf_3heads_w_sigmoid(4)

    model.to(device)
    model.eval()

    num_tests = 100
    times = []

    # input = torch.randn(1, 4).to(device) ###0.009654
    # input = torch.randn(256, 4).to(device) ### 0.009899  对应256 x 256 显存 1.6g
    # input = torch.randn(1200, 4).to(device) ### 0.011623  对应480 x 640 显存 2.3g
    input = torch.randn(9600, 4).to(device) ### 0.027910  对应1280 x 1920 显存 2.9g
    # 进行多次测量
    for _ in range(num_tests):
        start_time = time.perf_counter()  # 使用高精度计时器
        with torch.no_grad():  # 在推理时不需要计算梯度
            output = model(input)
        end_time = time.perf_counter()

        inference_time = end_time - start_time
        times.append(inference_time)
    # 计算平均推理时间
    average_time = np.mean(times)
    print(f"Average Inference Time: {average_time:.6f} seconds")