# Melanoma Classification

This repository contains code to create web application which use to detect melanome from given skin image.

## Introduction

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective. https://www.kaggle.com/c/siim-isic-melanoma-classification.

## Objective

The objective of this project is to identify melanoma in images of skin lesions. Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.In particular, we need to use images within the same patient and determine which are likely to represent a melanoma. In other words, we need to create a model which should predict the probability whether the lesion in the image is malignantor benign.Value 0 denotes benign, and 1 indicates malignant.

## DataSet 

The dataset which we are going to use are from following sources: 

Kaggle SIIM Melanoma Classification Challange :  https://www.kaggle.com/c/siim-isic-melanoma-classification

The dataset consists of images in :

  DIOCOM format
  
  JPEG format in JPEG directory
  
  TFRecord format in tfrecords directory

Additionally, there is a metadata comprising of train, test and submission file in CSV format.

## Exploritory Data Analysis : 
  
  The complete EDA of this dataset is available [here.](https://github.com/vatsal30/HackGujarat/tree/master/EDA)
  
## Model Used :
  
  In this project we used [ResNeXt50](https://github.com/facebookresearch/ResNeXt) which is pretrained on [Imagenet](http://www.image-net.org/).

## Training Process:
  
  For training we resized all the images into `224X224`.
  
  To convert all images into this fromat script is avaialable [here](https://github.com/vatsal30/HackGujarat/blob/master/resize_images.py).
  
  We used 10 fold StratifiedKfold and created new file which has KFlods. The script is avialable [here](https://github.com/vatsal30/HackGujarat/blob/master/create_folds.py)
  
  We used [train.py](https://github.com/vatsal30/HackGujarat/blob/master/train.py) to train this model on our dataset.
  
## Web App :
  `Streamlit` folder contains python script named [app.py](https://github.com/vatsal30/HackGujarat/blob/master/Streamlit/app.py) with a [Streamlit](https://www.streamlit.io/) app built around the model trained.
  and [prediction.py](https://github.com/vatsal30/HackGujarat/blob/master/Streamlit/prediction.py) contains predict function which takes an image and returns prediction.

## Hyperparameters 
  
  You can experiment with following hyperparametes to see different results:
  
  'resize_images.py' : image size
