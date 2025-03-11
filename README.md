# DCDI

### Towards Single-Lens Controllable Depth-of-Field Imaging via Depth-Aware Point Spread Functions [[PDF]](https://arxiv.org/pdf/2409.09754)
IEEE Transactions on Computational Imaging (TCI), 2025

Xiaolong Qian*, Qi Jiang*, Yao Gao, Shaohua Gao, Zhonghua Yi, [Lei Sun](https://ahupujr.github.io/), Kai Wei, Haifeng Li, [Kailun Yang](https://yangkailun.com/)†, [Kaiwei Wang](http://wangkaiwei.org/)†, [Jian Bai](https://person.zju.edu.cn/baijian)


Controllable Depth-of-Field (DoF) imaging commonly produces amazing visual effects based on heavy and expensive high-end lenses. However, confronted with the increasing demand for mobile scenarios, it is desirable to achieve a lightweight solution with Minimalist Optical Systems (MOS). This work centers around two major limitations of MOS, i.e., the severe optical aberrations and uncontrollable DoF, for achieving single-lens controllable DoF imaging via computational methods. A Depth-aware Controllable DoF Imaging (DCDI) framework is proposed equipped with All-in-Focus (AiF) aberration correction and monocular depth estimation, where the recovered image and corresponding depth map are utilized to produce imaging results under diverse DoFs of any high-end lens via patch-wise convolution.

The source code will be made publicly available after the paper is accepted.
![image](https://github.com/XiaolongQian/DCDI/blob/main/real_world_result.png)

### Installation:
The implementation of our work is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), which is an open source toolbox for image/video restoration tasks.
```
conda create -n depthmos python=3.8
conda activate depthmos
conda install cudatoolkit==11.6 -c nvidia
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
python setup.py develop
```
### Training:
```
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/dacn/train_dacn.yml
```

### License:
This project is under the MIT license, and it is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) which is under Apache 2.0 license.

### 🤝 Publication:
Please consider referencing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```
@article{qian2025dcdi,
  title={Towards Single-Lens Controllable Depth-of-Field Imaging via Depth-Aware Point Spread Functions},
  author={Xiaolong Qian and Qi Jiang and Yao Gao and Shaohua Gao and Zhonghua Yi and Lei Sun and Kai Wei and Haifeng Li and Kailun Yang and Kaiwei Wang and Jian Bai},
  journal={IEEE Transactions on Computational Imaging},
  year={2025}
}
```
