import os
import torch
import matplotlib.pyplot as plt
import numpy as np

import cv2
import torch.nn as nn

from tqdm import tqdm
from os import replace
from numpy.core.numeric import Inf
from FOD.utils import get_losses, get_optimizer, get_schedulers, create_dir
from FOD.FocusOnDepth import FocusOnDepth

class Trainer(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = self.config['General']['type']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize_h = config['Dataset']['transforms']['resize_h']
        resize_w = config['Dataset']['transforms']['resize_w']
        self.model = FocusOnDepth(
                    image_size  =   (3,resize_h,resize_w),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    nclasses    =   len(config['Dataset']['classes']) + 1,
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['General']['patch_size'],
        )

        self.model.to(self.device)
        self.load_loss = Inf
        # print(self.model)
        # exit(0)

        self.loss_depth, self.loss_segmentation = get_losses(config)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model)
        self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch])

    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']

        val_loss = min(Inf,self.load_loss)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(self.config, self.model)
        self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch])
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Epoch ", epoch+1)
            running_loss = 0.0
            self.model.train()
            pbar = tqdm(train_dataloader)
            pbar.set_description("Training")
            for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
                # get the inputs; data is a list of [inputs, labels]
                X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(self.device), Y_segmentations.to(self.device)
                # zero the parameter gradients
                self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()
                # forward + backward + optimizer
                output_depths, output_segmentations = self.model(X)
                output_depths = output_depths.squeeze(1) if output_depths != None else None

                Y_depths = Y_depths.squeeze(1) #1xHxW -> HxW
                Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss
                loss = self.loss_depth(output_depths, Y_depths) + self.loss_segmentation(output_segmentations, Y_segmentations)
                loss.backward()
                # step optimizer
                self.optimizer_scratch.step()
                self.optimizer_backbone.step()

                running_loss += loss.item()
                if np.isnan(running_loss):
                    print('\n',
                        X.min().item(), X.max().item(),'\n',
                        Y_depths.min().item(), Y_depths.max().item(),'\n',
                        output_depths.min().item(), output_depths.max().item(),'\n',
                        loss.item(),
                    )
                    exit(0)


            new_val_loss = self.run_eval(val_dataloader)

            if new_val_loss < val_loss:
                self.save_model(new_val_loss)
                val_loss = new_val_loss

            self.schedulers[0].step(new_val_loss)
            self.schedulers[1].step(new_val_loss)

        print('Finished Training')

    def run_eval(self, val_dataloader):
        """
            :- val_dataloader -: torch dataloader
        """
        val_loss = 0.
        self.model.eval()
        X_1 = None
        Y_depths_1 = None
        Y_segmentations_1 = None
        output_depths_1 = None
        output_segmentations_1 = None
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
                X, Y_depths, Y_segmentations = X.to(self.device), Y_depths.to(self.device), Y_segmentations.to(self.device)
                output_depths, output_segmentations = self.model(X)
                output_depths = output_depths.squeeze(1) if output_depths != None else None
                Y_depths = Y_depths.squeeze(1)
                Y_segmentations = Y_segmentations.squeeze(1)
                if i==0:
                    X_1 = X
                    Y_depths_1 = Y_depths
                    Y_segmentations_1 = Y_segmentations
                    output_depths_1 = output_depths
                    output_segmentations_1 = output_segmentations
                # get loss
                loss = self.loss_depth(output_depths, Y_depths) + self.loss_segmentation(output_segmentations, Y_segmentations)
                val_loss += loss.item()
                pbar.set_postfix({'validation_loss': val_loss/(i+1)})

        return val_loss/(i+1)

    def save_model(self,loss):
        # path_model = os.path.join(self.config['General']['path_model'], self.model.__class__.__name__)
        path_model = os.path.join(self.config['General']['path_model'], 'FocusOnDepth_{}.p'.format(self.config['General']['model_timm']))
        create_dir(path_model)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                    'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict(),
                    'loss' : loss
                    }, path_model+'.p')
        print('Model saved at : {}'.format(path_model))

    def img_logger(self, X, Y_depths, Y_segmentations, output_depths, output_segmentations):
        tmp = X[:nb_to_show].detach().cpu().numpy()
        imgs = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        if output_depths != None:
            tmp = Y_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            depth_truths = np.repeat(tmp, 3, axis=1)
            tmp = output_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            tmp = np.repeat(tmp, 3, axis=1)
            #depth_preds = 1.0 - tmp
            depth_preds = tmp
        if output_segmentations != None:
            tmp = Y_segmentations[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            segmentation_truths = np.repeat(tmp, 3, axis=1).astype('float32')
            tmp = torch.argmax(output_segmentations[:nb_to_show], dim=1)
            tmp = tmp.unsqueeze(1).detach().cpu().numpy()
            tmp = np.repeat(tmp, 3, axis=1)
            segmentation_preds = tmp.astype('float32')
        print("******************************************************")
        print(imgs.shape, imgs.mean().item(), imgs.max().item(), imgs.min().item())
        if output_depths != None:
            print(depth_truths.shape, depth_truths.mean().item(), depth_truths.max().item(), depth_truths.min().item())
            print(depth_preds.shape, depth_preds.mean().item(), depth_preds.max().item(), depth_preds.min().item())
        # if output_segmentations != None:
        #     print(segmentation_truths.shape, segmentation_truths.mean().item(), segmentation_truths.max().item(), segmentation_truths.min().item())
        #     print(segmentation_preds.shape, segmentation_preds.mean().item(), segmentation_preds.max().item(), segmentation_preds.min().item())
        print("******************************************************")
        imgs = imgs.transpose(0,2,3,1)
        if output_depths != None:
            depth_truths = depth_truths.transpose(0,2,3,1)
            depth_preds = depth_preds.transpose(0,2,3,1)
        if output_segmentations != None:
            segmentation_truths = segmentation_truths.transpose(0,2,3,1)
            segmentation_preds = segmentation_preds.transpose(0,2,3,1)

    def load_checkpoint(self, path_model):
        if os.path.isfile(path_model):
            checkpoint = torch.load(path_model, map_location='cpu')
            if self.model is not None and checkpoint['model_state_dict'] is not None:
                self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError
        self.load_loss = checkpoint['loss']