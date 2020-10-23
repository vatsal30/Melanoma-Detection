import os
import torch
import pretrainedmodels
import albumentations

import numpy as np 
import pandas as pd 
from sklearn import metrics

import torch.nn as nn
import torch.nn.functional as F

from wtfml.data_loaders.image import ClassificationLoader
# from wtfml.utils import EarlyStopping
from wtfml.engine import Engine

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained = pretrained)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1,1).type_as(x)
        )
        return out, loss

def predict(image_path, fold):

    device="cpu"
    model_path = "../model/"

    mean =  (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationLoader(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4
    )

    model = SEResNext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(os.path.join(model_path, f"model_fold_{fold}.bin"))) #if gets error here add map_location=torch.device('cpu')

    predictions =  Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()
    print(predictions)
    return predictions[0]

def ensemble(image_path):
    prediction = [predict(image_path, fold) for fold in range(5)]
    print(prediction)
    prediction = sum(prediction)/5
    return prediction
