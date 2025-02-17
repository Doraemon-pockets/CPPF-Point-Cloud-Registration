# Enhanced Point Pair Features for Point Cloud Registration

## Abstract

Point cloud registration is a crucial task in 3D computer vision, but current methods face challenges such as complex geometric shapes, noise, and missing data in the target cloud. These challenges often lead to time-consuming point pair feature extraction and insufficient extraction of effective features, thereby reducing registration efficiency. To address this, our research introduces a method that combines adaptive downsampling and curvature enhancement for point pair feature extraction, aiming to achieve efficient point cloud registration. Our method utilizes adaptive downsampling based on normal angle perception to balance information preservation and computational complexity. We propose the Curvature-Enhanced Point Pair Feature (CPPF) module to enhance the capture of local features, improving sensitivity to geometric changes. The attention mechanism module integrates local and global features, enhancing representational capacity and information richness. Experiments conducted on ModelNet40 and real industrial components demonstrate that our method performs well across various scenarios, surpassing existing methods.

![](./common/Network.jpg)

## Environment

- requirements.txt `pip install -r requirements.txt`
- open3d-python==0.9.0.0 `python -m pip install open3d==0.9`

## Dataset

Download [ModelNet40](https://modelnet.cs.princeton.edu) from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) [435M].


## Model Training

```bash
mkdir cppfnet && cd cppfnet
git clone git@github.com:Doraemon-pockets/CPPF-Point-Cloud-Registration.git
python train.py --noise_type crop
```


For clean data, we use a batch size of 8, but you can also adjust it dynamically based on your GPU configuration.
```bash
python train.py --noise_type clean --train_batch_size 8
```

, and for noisy data:

```bash
python train.py --noise_type jitter --train_batch_size 8
```
, and for partial data:

```bash
python train.py --noise_type crop --train_batch_size 8
```

## Inference / Evaluation

This script performs inference on the trained model, and computes evaluation metrics.


```bash
python eval.py --noise_type clean --resume [path-to-logs/ckpt/model-best.pth]
```

##Registration Visualization

```bash
python vis.py --noise_type crop --resume [path-to-model.pth] --dataset_path [your_path]
```
