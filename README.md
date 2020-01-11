# KinD++
This is a Tensorflow implementation of KinD++. (Beyond Brightening Low-light Images)

We propose a novel multi-scale illumination attention module (MSIA), which can alleviate visual defects (e.g. non-uniform spots and over-smoothing) left in [KinD](https://github.com/zhangyhuaee/KinD). 

The KinD net was proposed in the following [Paper](http://doi.acm.org/10.1145/3343031.3350926).

Kindling the Darkness: a Practical Low-light Image Enhancer. In ACM MM 2019<br>
Yonghua Zhang, Jiawan Zhang, Xiaojie Guo

### The network architecture of KinD++: ###
<img src="figures/network4.jpg" width="1000px"/>

### The reflectance restoration network and the MSIA module: ###
<img src="figures/restoration_net2.jpg" width="1000px"/> 

### Visual comparison with state-of-the-art low-light image enhancement methods. ###
<img src="figures/compare_images.jpg" width="1000px"/>

### Requirements ###
1. Python
2. Tensorflow >= 1.10.0
3. numpy, PIL

### Test ###
Please put test images into 'test_images' folder and download the pre-trained checkpoints from [google drive](https://drive.google.com/open?id=1RuW6fgkDEQ6v9GMlcWgtWiGglew6jplO), then just run
```shell
python evaluate.py
```
The test datasets (e.g. DICM, LIME, MEF and NPE) can be downloaded from [google drive](https://drive.google.com/open?id=12sUp8aOlNIB5h11lwsjs1Qm9sdH7v5p1).

### Train ###
The original LOLdataset can be downloaded from [here](https://daooshee.github.io/BMVC2018website/). We rearrange the original LOLdataset and add several pairs all-zero images and 260 pairs synthetic images to improve the decomposition results and restoration results. The training dataset can be download from [google drive](https://drive.google.com/open?id=1YztDWbK3MV5EroSpuWmYlPsmFcFGoLmq). For training, just run
```shell
python decomposition_net_train.py
python illumination_adjustment_net_train.py
python reflectance_restoration_net_train.py
```
You can also evaluate on the LOLdataset, just run
```shell
python evaluate_LOLdataset.py
```

### More information ###
We will provide more codes link of the existing low-light image enhancement methods in the few days. So stay tuned! 



### NIQE code ###
Non-reference metric NIQE are adopted for quantitative comparison. The original code for computing NIQE is [here](https://github.com/csjunxu/Bovik_NIQE_SPL2013). To improve the robustness, we follow the author's code and retrain the model parameters by extending 100 high-resolution natural images from [PIRM dataset](https://pirm.github.io/).  Put the [original 125 images](http://live.ece.utexas.edu/research/quality/pristinedata.zip) and additional 100 images (dir: PIRM_dataset\Validation\Original) into one folder 'data', then run
```shell
[mu_prisparam cov_prisparam]  = estimatemodelparam('data',96,96,0,0,0.75);
```
After retrained, the file 'modelparameters_new.mat' will be generated. We use this model to evaluate all results.

### References ###
[1] Y. Zhang, J. Zhang, and X. Guo, “Kindling the darkness: A practical low-light image enhancer,” in ACM MM, 2019, pp. 1632–1640.

[2] X. Guo, Y. Li, and H. Ling, “Lime: Low-light image enhancement via illumination map estimation,” IEEE TIP, vol. 26, no. 2, pp. 982–993, 2017.

[3] C. Wei, W. Wang, W. Yang, and J. Liu, “Deep retinex decomposition for low-light enhancement,” in BMVC, 2018.

[4] R. Wang, Q. Zhang, C.-W. Fu, X. Shen, W.-S. Zheng, and J. Jia, “Underexposed photo enhancement using deep illumination estimation,” in CVPR, 2019, pp. 6849–6857.


