# -*- coding: utf-8 -*-
"""BDMH Projet_Squeeznet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DMTq13KLJZ5hwEIdqesW4Tk97jtdRb5j
"""

## Cell 0 ##
##Import all libraries


from fastai import *
from fastai.vision import *

## Cell 1 ##
## Define path of images ##


import glob
import os
Labels=[]
for filename in glob.glob("drive/My Drive/kai/*"):
  if not os.path.isdir(filename):
    Labels.append(int(filename[19]))
    print(filename)
print(Labels)

path = "drive/My Drive/kai/"
print(path)

fn_paths = []

for filename in glob.glob("drive/My Drive/kai/*"):
  if not os.path.isdir(filename):
    fn_paths.append(filename)

print(fn_paths)

##Cell 2 ##
## Prepare Data using image data bunch and # Creation of transformation object


data = ImageDataBunch.from_lists(path, fn_paths,labels=Labels,  size=256, bs=20)
data.normalize(imagenet_stats)

## Cell3 ##
## Download pre trained model ##


learn=cnn_learner(data,models.squeezenet1_1  ,metrics=accuracy)

## Cell 4 ##
## Find optimal learning Rates ##


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion =True)

## Cell 5##
## Train Model with optimal Learning Rates ##

learn.fit_one_cycle(5,slice(1e-6,1e-5 ))

## Cell 6 ##
## Load model using the saved state ##


learn.load("step_s_2")

## Cell 7 ##
## Draw Confusion Matrix ##


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(5,5))

## Cell 8 ##
## Draw Summary of Evaluation Parameters ##


preds, y_true=learn.get_preds(ds_type=DatasetType.Valid)
y_true=y_true.numpy() 
preds=np.argmax(preds.numpy(), axis=-1)
from sklearn.metrics import auc, roc_curve, precision_recall_curve, classification_report

report = classification_report(y_true, preds, target_names=['1','0'])
print(report)

## Cell 9 ##

## Misclassified Images with top loss ##


interp.plot_top_losses(9, figsize=(8,8))

interp.most_confused(min_val=2)