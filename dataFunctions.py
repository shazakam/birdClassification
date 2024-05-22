import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
import os
from torchvision import transforms
import torchvision.transforms as transforms

from torchvision.models import ResNet34_Weights

class BirdImageDataset(Dataset):
    
    def __init__(self, annotations_file, train):
        target_size = (224, 224)
        def change_path(filepath):
            els = filepath.split('/')
            
            if train:
                els[1] = 'PARAKETT  AUKLET'
            else:
                els[1] = 'PARAKETT AUKLET'
            return os.path.join(*els)
    
        self.img_labels = pd.read_csv(annotations_file)
        # self.transform = transforms.Compose([
        #         transforms.Resize(target_size)
        #     ])

        self.weights = ResNet34_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.img_labels.loc[self.img_labels['labels'] == 'PARAKETT  AKULET', 'filepaths'] =  self.img_labels.loc[self.img_labels['labels'] == 'PARAKETT  AKULET', 'filepaths'].map(change_path)
        
        

    def __len__(self):
        return self.img_labels.shape[0]

    def __get__unprocessed_item__(self, idx):
        img_path = '100-bird-species/'+self.img_labels['filepaths'].iloc[idx]
        image = read_image(img_path)/255
        # image = self.transform(image)
        # image = self.preprocess(read_image(img_path))
        label = torch.zeros(525)
        label[int(self.img_labels['class id'].iloc[idx])] = 1
        return image, label

    def __getitem__(self, idx):
        img_path = '100-bird-species/'+self.img_labels['filepaths'].iloc[idx]
        # image = read_image(img_path)/255
        # image = self.transform(image)
        image = self.preprocess(read_image(img_path))
        label = torch.zeros(525)
        label[int(self.img_labels['class id'].iloc[idx])] = 1
        return image, label

    def __get_image_from_species__(self, species_label):
        first_elem = self.img_labels[self.img_labels['labels'] == species_label]['filepaths'].iloc[0]
        img_path = '100-bird-species/'+first_elem
        image = read_image(img_path)/255

        return image