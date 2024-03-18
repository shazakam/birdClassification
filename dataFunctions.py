import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import os
from torchvision import transforms
import torchvision.transforms as transforms



class BirdImageDataset(Dataset):
    
    def __init__(self, annotations_file):
        target_size = (224, 224)
        def change_path(filepath):
            els = filepath.split('/')
            els[1] = 'PARAKETT  AUKLET'
            return os.path.join(*els)
    
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transforms.Compose([
                transforms.Resize(target_size)
            ])
        self.img_labels.loc[self.img_labels['labels'] == 'PARAKETT  AKULET', 'filepaths'] =  self.img_labels.loc[self.img_labels['labels'] == 'PARAKETT  AKULET', 'filepaths'].map(change_path)
        
        

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, idx):
        img_path = self.img_labels['filepaths'].iloc[idx]
        image = read_image(img_path)/255
        image = self.transform(image)
        label = torch.zeros(525)
        label[int(self.img_labels['class id'].iloc[idx])] = 1
        return image, label