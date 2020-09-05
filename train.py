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
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine

from apex import amp

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained = pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1,1).type_as(out)
        )
        return out, loss
    
def train(fold):

    training_data_path = "/media/vatsal/Movies & Games/down_siim-isic-melanoma-classification/train224/"
    model_path = f"/media/vatsal/Movies & Games/Melenoma-Deep-Learning/model/"
    df = pd.read_csv("/media/vatsal/Movies & Games/down_siim-isic-melanoma-classification/train_folds.csv")
    device = "cuda"
    epochs = 50
    train_bs = 8
    valid_bs = 8
    mean =  (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )


    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )
    
    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i+ ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i+ ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=4
    )
    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(patience=5, mode="max")

    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level="O1",
        verbosity=0
    )

    for epoch in range(epochs):
        
        train_loss = Engine.train(train_loader, model, optimizer, device=device, fp16=True)
        
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=device
        )
        print(predictions)
        
        predictions = np.vstack((predictions)).ravel()
        
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, os.path.join(model_path, f"model{fold}.bin"))
        if es.early_stop:
            print('early stopping')
            break

def predict(fold):

    training_data_path = "/media/vatsal/Movies & Games/down_siim-isic-melanoma-classification/test224/"
    model_path = "/media/vatsal/Movies & Games/Melenoma-Deep-Learning/model/"
    df_test = pd.read_csv("/media/vatsal/Movies & Games/down_siim-isic-melanoma-classification/test.csv")
    df_test.loc[:,"target"] = 0

    device = "cuda"
    epochs = 50
    test_bs = 8
    mean =  (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = df_test.image_name.values.tolist()
    test_images = [os.path.join(training_data_path, i+ ".jpg") for i in test_images]
    test_targets = df_test.target.values

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
    model.load_state_dict(torch.load(os.path.join(model_path, f"model{fold}.bin")))
    model.to(device)


    predictions =  Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()
    return predictions


if __name__=="__main__":
    train(fold=0)
