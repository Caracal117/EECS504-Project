# EECS 504 course project - Monocular Depth Estimation with Vision Transformer

## Abstract

The project focuses on advancing monocular depth estimation using Vision Transformers (ViT). Unlike traditional methods that rely on geometry or conventional learning techniques, our approach leverages the cutting-edge ViT for a more comprehensive understanding of depth in single images. This method involves training the model to assign depth values to each pixel, representing the distance from the camera. Our work in this project is to reimplement the DPT neural network and adding training, evaluation work to make the paper content feasible to training on any dataset.

## Requirements

``` pip install -r requirements.txt ```

## Running the model

You can first download one of the models:

### Pretrained Model

Get the links of the following model:

+ [```Model trained on NYU2 Dataset```](https://drive.google.com/file/d/1NvVQr5WoMRDrhnNc3tbhc43pYrPZMp0T/view?usp=drive_link)

And put the ```.p``` file into the directory ```models/```. After that, you need to update the ```config.json``` according to the pre-trained model you have chosen to run the predictions (this means that if you load a depth-only model, then you have to set ```type``` to ```depth``` for example ...).

### Run a prediction

Put your input images (that have to be ```.png``` or ```.jpg```) into the ```input/``` folder. Then, just run ```python run.py``` and you should get the depth maps as well as the segmentation masks in the ```output/``` folder.

## Training

### Build the dataset

Our model is got from Kaggle
+ [NYU2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | [view on Kaggle](https://www.kaggle.com/antocad/nyuv2-fod)

### Run the training script
Run the training script: ```python train.py```

## Citations
Our work is based on work from [Vision Transformers for Dense Prediction](https://github.com/isl-org/DPT) and [FocusOnDepth](https://github.com/antocad/FocusOnDepth)

```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```

```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```
