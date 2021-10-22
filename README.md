# Kinetic-GAN
This repository contains the official PyTorch implementation of the following paper:
> **Generative Adversarial Graph Convolutional Networks for Human Action Synthesis**, Bruno Degardin, João Neves, Vasco Lopes, João Brito, Ehsan Yaghoubi and Hugo Proença, WACV 2022. [[Arxiv Preprint]](https://arxiv.org/abs/2110.11191)

## Resources

Material related to our paper is available via the following links:

- Paper: https://arxiv.org/abs/2110.11191
- Video: TBA
- Code: https://github.com/DegardinBruno/Kinetic-GAN
- Datasets
  - NTU RGB+D: http://socia-lab.di.ubi.pt
  - NTU-120 RGB+D: http://socia-lab.di.ubi.pt
  - NTU-2D RGB+D: http://socia-lab.di.ubi.pt
  - Human3.6M: http://socia-lab.di.ubi.pt


## Model Zoo and Benchmarks

PyTorchVideo provides reference implementation of a large number of video understanding approaches. In this document, we also provide comprehensive benchmarks to evaluate the supported models on different datasets using standard evaluation setup. All the models can be downloaded from the provided links.

### NTU RGB+D

arch     | benchmark | frame length | FID | Config | Model
-------- | --------- | ------------ | --- | ------ | -----
kinetic-gan-mlp4 | cross-subject | 64 | 3.618 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)
kinetic-gan-mlp6 | cross-view | 64 | 4.235 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)

### Download NTU RGB+D X-Subject Dataset
FileZilla -> IP: 10.0.4.137 -> video -> needed_DEGARDIN -> DATASETS -> NTU-RGBD -> xsub -> train_data.npy and train_label.pkl


### Kinetic-GAN
Kinetic-GAN with NTU RGB+D xsub dataset as input

#### Train
```
python kinetic-gan.py --data_path path_train_data --label_path path_train_labels
```

---

### Visualization
Visualization of synthetic samples. Check [NTU-RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset labels (index=label-1).
Generator synthesizes 10 samples from the 60 classes. Classes are repeated at every 60 samples, check "jump up" example below.

```
python visualization/synthetic.py --path path_samples --index_sample 26 86 146  # Multiple samples indexes (Max 3) 
```

Check current loss.
```
python visualization/plot_loss.py  # Check inside settings
```
