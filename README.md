# EECS 504 course project - Monocular Depth Estimation with Vision Transformer


<!-- ![presentation](https://i.ibb.co/rbySmMc/DL-FOD-POSTER-1.png) -->

## Abstract

The project focuses on advancing monocular depth estimation using Vision Transformers (ViT). Unlike traditional methods that rely on geometry or conventional learning techniques, our approach leverages the cutting-edge ViT for a more comprehensive understanding of depth in single images. This method involves training the model to assign depth values to each pixel, representing the distance from the camera. Our work in this project is to reimplement the DPT neural network and adding training, evaluation work to make the paper content feasible to training on any dataset.

## Requirements

``` pip install -r requirements.txt ```

## Running the model

You can first download one of the models from the model zoo:

### Pretrained Model

Get the links of the following models:

+ [```FocusOnDepth_vit_base_patch16_384.p```](https://drive.google.com/file/d/1Q7I777FW_dz5p5UlMsD6aktWQ1eyR1vN/view?usp=sharing)
+ Other models coming soon...

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

```
@article{DPT,
  author    = {Ren{\'{e}} Ranftl and
               Alexey Bochkovskiy and
               Vladlen Koltun},
  title     = {Vision Transformers for Dense Prediction},
  journal   = {CoRR},
  volume    = {abs/2103.13413},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.13413},
  eprinttype = {arXiv},
  eprint    = {2103.13413},
  timestamp = {Wed, 07 Apr 2021 15:31:46 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-13413.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```