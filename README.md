# Differencing based Self-supervised pretraining for scene change detection (DSP)


**This is the official code for COLLA 2022 Paper, ["Differencing based Self-supervised pretraining for scene change detection"](paper) by [Vijaya Raghavan Thiruvengadathan Ramkumar](https://www.linkedin.com/in/vijayaraghavan95), [Elahe Arani](https://www.linkedin.com/in/elahe-arani-630870b2/) and [Bahram Zonooz](https://www.linkedin.com/in/bahram-zonooz-2b5589156/), where we propose a novel self-supervised pretraining architechture based on differenceing called DSP for scene change detection.**

## Abstract


Scene change detection (SCD), a crucial perception task, identifies changes by comparing scenes captured at different times. SCD is challenging due to noisy changes in illumination, seasonal variations, and perspective differences across a pair of views. Deep neural network based solutions require a large quantity of annotated data which is tedious and expensive to obtain. On the other hand, transfer learning from large datasets induces domain shift. To address these challenges, we propose a novel Differencing self-supervised pretraining (DSP) method that uses feature differencing to learn discriminatory representations corresponding to the changed regions while simultaneously tackling the noisy changes by enforcing temporal invariance across views. Our experimental results on SCD datasets demonstrate the effectiveness of our method, specifically to differences in camera viewpoints and lighting conditions. Compared against the self-supervised Barlow Twins and the standard ImageNet pretraining that uses more than a million additional labeled images, DSP can surpass it without using any additional data. Our results also demonstrate the robustness of DSP to natural corruptions, distribution shift, and learning under limited labeled data.

![alt text](https://github.com/NeurAI-Lab/DSP/blob/main/method.png)

For more details, please see the [Paper](https://arxiv.org/abs/2208.05838) and [Presentation](https://www.youtube.com/watch?v=kWUxxC5hjKw).

## Requirements

- python 3.6+
- opencv 3.4.2+
- pytorch 1.6.0
- torchvision 0.4.0+
- tqdm 4.51.0
- tensorboardX 2.1

## Datasets

Our network is tested on two datasets for street-view scene change detection. 

- 'PCD' dataset from [Change detection from a street image pair using CNN features and superpixel segmentation](http://www.vision.is.tohoku.ac.jp/files/9814/3947/4830/71-Sakurada-BMVC15.pdf). 
  - You can find the information about how to get 'TSUNAMI', 'GSV' and preprocessed datasets for training and test [here](https://kensakurada.github.io/pcd_dataset.html).
- 'VL-CMU-CD' dataset from [Street-View Change Detection with Deconvolutional Networks](http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf).
  -  'VL-CMU-CD': [[googledrive]](https://drive.google.com/file/d/0B-IG2NONFdciOWY5QkQ3OUgwejQ/view?resourcekey=0-rEzCjPFmDFjt4UMWamV4Eg)

## Dataset Preprocessing

- For DSP pretraining - included in the DSP--dataset--CMU.py/PCD.py
- For finetuning and evaluation - Please follow the preprocessing method used by the official implementation of [{Dynamic Receptive Temporal Attention Network for Street Scene Change Detection paper}](https://github.com/Herrccc/DR-TANet) 

Dataset folder structure for VL-CMU-CD:
```bash
├── VL-CMU-CD
│   ├── Image_T0
│   ├── Image_T1
│   ├── Ground Truth

```
								
## SSL Training


- For training 'DSP' on VL-CMU-CD dataset:
```
python3 DSP/train.py --ssl_batchsize 16 --ssl_epochs 500 --save_dir /outputs --data_dir /path/to/VL-CMU-CD --img_size 256 --n_proj 256 --hidden_layer 512 --output_stride 8 --pre_train False --m_backbone False --barlow_twins True --dense_cl False --kd_loss True --kd_loss_2 sp --inter_kl False --alpha_inter_kd 0 --alpha_sp 3000 --alpha_kl 100
```
 

## Fine Tuning

We evaluate Rand, Imagenet supervised, Barlow twins, and DSP pretraining on DR-TANet.
- Follow the Please follow the train and test procedure used by the official implementation of [{Dynamic Receptive Temporal Attention Network for Street Scene Change Detection paper}](https://github.com/Herrccc/DR-TANet) 

Start training with DR-TANet on 'VL-CMU-CD' dataset.

    python3 train.py --dataset vl_cmu_cd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --max-epochs 150 --batch-size 16 --encoder-arch resnet50 --epoch-save 25 --drtam --refinement

Start evaluating with DR-TANet on 'PCD' dataset.

    python3 eval.py --dataset pcd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --resultdir /path_to_save_eval_result --encoder-arch resnet50 --drtam --refinement --store-imgs
  
## Evaluating the finetuned model

Start evaluating with DR-TANet on 'PCD' dataset.

    python3 eval.py --dataset pcd --datadir /path_to_dataset --checkpointdir /path_to_check_point_directory --resultdir /path_to_save_eval_result --encoder-arch resnet18 --drtam --refinement --store-imgs
    
## Analysis
We analyse our DSP model under 3 scenarios: **1. Robustness to Natural corruptions 2. Out-of-distribution data 3. Limited labeled data. For more details, please see the [Paper](https://arxiv.org/abs/2208.05838).** 
For Natural corruptions evaluation, please refer to the paper [{Benchmarking Neural Network Robustness to
Common Corruptions and Surface Variations }](https://arxiv.org/pdf/1807.01697.pdf) 

And finally, for the ease of comparison, we have provided the model checkpoints for the DSP pretraining below:  [google drive](https://drive.google.com/drive/folders/1UwFQ7NjXRwyfgfhFnX6_CPTm8hQ8AoFF?usp=sharing)


## Cite our work

If you find the code useful in your research, please consider citing our paper:

<pre>
@article{ramkumar2022differencing,
  title={Differencing based Self-supervised pretraining for Scene Change Detection},
  author={Ramkumar, Vijaya Raghavan T and Arani, Elahe and Zonooz, Bahram},
  journal={arXiv preprint arXiv:2208.05838},
  year={2022}
}
