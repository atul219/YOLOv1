import argparse
import os
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils import *




ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help = 'path of dataset directory')

args = vars(ap.parse_args())

print("Dataset file {}".format(args['dataset']))

class VOCDataset(Dataset):
    def __init__(self, data_dir, 
                img_dir, 
                label_dir, 
                train_csv, 
                split_size = 7,
                num_boxes = 2,
                num_classes = len(classes_num),
                transform = None):
        

        self.data_dir = data_dir
        self.img_dir = img_dir
        self.label__dir = label_dir
        self.csv = train_csv
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.classes = num_classes
        self.tfms = transform
    

    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.csv.iloc[idx, 1])
        img = read_image(img_path)

        # # label
        label_path = os.path.join(self.label__dir, self.csv.iloc[idx, 1])

        boxes = []

        with open(label_path) as f:
            for l in f.readlines():
                label = []
                for x in l.replace('\n', "").split():
                    
                    # to check if its class label or bounding box coordinates
                    # if x is float then its a bounding box coordinates else its class label
                    if float(x) != int(float(x)):
                        x = float(x)
                    else:
                        x = int(x)
                    
                    label.append(x)
            

                boxes.append(label)
        

        # check if any transforms:
        if self.tfms:
            img, boxes = self.tfms(img, boxes)
        
        # Convert to cells
        label_mat = torch.zeros(self.split_size, 
                                self.split_size,
                                30) # 30 is total classes + object score  + 4 bounding boxes


        for b in boxes:
            cls_label, x, y, w, h = b

            a,b = int(self.split_size * y), int(self.split_size * x)
            x_cell, y_cell = (self.split_size * x - b), (self.split_size * y - a)

            w_cell, h_cell = (w*self.split_size,  h*self.split_size)

            if label_mat[a,b, 20] == 0: # if no object found already

                label_mat[a,b,20] == 1

                box_coord = [x_cell, y_cell, w_cell, h_cell]

                label_mat[a,b, 21:25] = box_coord

                label_mat[a,b, cls_label] = 1
            
        
        return img, label_mat