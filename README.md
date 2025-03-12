# DCDI

### Towards Single-Lens Controllable Depth-of-Field Imaging via Depth-Aware Point Spread Functions [[PDF]](https://arxiv.org/pdf/2409.09754)
IEEE Transactions on Computational Imaging (TCI), 2025

Xiaolong Qian*, Qi Jiang*, Yao Gao, Shaohua Gao, Zhonghua Yi, [Lei Sun](https://ahupujr.github.io/), Kai Wei, Haifeng Li, [Kailun Yang](https://yangkailun.com/)†, [Kaiwei Wang](http://wangkaiwei.org/)†, [Jian Bai](https://person.zju.edu.cn/baijian)


Controllable Depth-of-Field (DoF) imaging commonly produces amazing visual effects based on heavy and expensive high-end lenses. However, confronted with the increasing demand for mobile scenarios, it is desirable to achieve a lightweight solution with Minimalist Optical Systems (MOS). This work centers around two major limitations of MOS, i.e., the severe optical aberrations and uncontrollable DoF, for achieving single-lens controllable DoF imaging via computational methods. A Depth-aware Controllable DoF Imaging (DCDI) framework is proposed equipped with All-in-Focus (AiF) aberration correction and monocular depth estimation, where the recovered image and corresponding depth map are utilized to produce imaging results under diverse DoFs of any high-end lens via patch-wise convolution.

The source code will be made publicly available after the paper is accepted.
![image](https://github.com/XiaolongQian/DCDI/blob/main/real_world_result.png)
### Overview
![image](https://github.com/XiaolongQian/DCDI/blob/main/figures/framework.png)
The proposed DCDI framework is illustrated in Fig. 2. Our framework is comprised of four components: the simulation of a depth-aware aberration dataset, an MDE module, a CAC module, and a PSF representation module. In Sec. III-A, we
introduce a method for simulating a depth-aware aberration dataset, which serves as a foundation for enabling the network to adaptively learn depth-varying degradation characteristics during Depth-aware Degradation-adaptive Training (DA2T) scheme. In Sec. III-B, Residual Depth-Image Cross-Attention Block (RDICAB) and Depth-aware Deformable Convolution Block (D2CB) are proposed as two depth-aware mechanisms to enhance the recovery performance of the Depth-Aware
Correction Network (DACN). In Sec. III-C, we design the Omni-Lens-Field to represent the 4D PSFLib of various lenses. Based on the recovered AiF aberration-free image, the depth map estimated by UniDepth [11] and the depth-aware PSF
map inferred by Omni-Lens-Field, single-lens controllable DoF imaging is achieved.
### Results
![image](https://github.com/XiaolongQian/DCDI/blob/main/figures/DA2T.png)
Comparison of recovery results of different training schemes. For comparison purposes, we first simulate a depth-unaware training dataset as our baseline, which is obtained by convolving the PSF map of the same scene depth. To
verify the generalization of our proposed DA2T scheme, we select the main components of four representative state-of-the-art SR models, including a CNN-based Module, i.e., Residual Block (RB) in EDSR [23] and three transformer-based Modules including Residual Swin Transformer Block (RSTB) in SwinIR [24], Permuted Self-Attention (PSA) in SRformer [25], and Residual Deep-feature-extraction Group (RDG) in DRCT [26]) for our experiments. To ensure a fair comparison, the experimental settings of the two training schemes are the same, including model parameters, initial learning rate, decay strategy, and number of training epochs.
![image](https://github.com/XiaolongQian/DCDI/blob/main/figures/Depth_mechanism.png)
Restoration experiment of depth-aware mechanism. Similar to the experiment in the previous section, we sequentially replaced the RACM in DACN with four representative stateof-the-art super-resolution models, to investigate the recovery capacity of our proposed depth-aware mechanisms (RDICAB and D2CB).
![image](https://github.com/XiaolongQian/DCDI/blob/main/figures/outdoor_real_wolrd.png)
Outdoor Scene Evaluation. The qualitative results of outdoor scenes are displayed in Fig. 10. Real-world outdoor scenes, characterized by more dynamic depth variations, present greater challenges for restoration.
![image](https://github.com/XiaolongQian/DCDI/blob/main/figures/controllable DOF imaging.png)
The real-world outdoor controllable DoF image results are presented in Fig. 13.
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
### Concat:
Please concat me via ```xiaolongqian@zju.edu.cn``` if you have any questions.
