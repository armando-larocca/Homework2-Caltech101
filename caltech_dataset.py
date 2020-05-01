from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        f=open(root+split+'.txt', "r")
       
        dataset = []
        for line in f:
            if(line.find("BACKGROUND_Google")==-1):              
                dataset.append(line)

        f.close()
        print(len(dataset))
              
        labels = os.listdir("Caltech101/101_ObjectCategories")
        labels.remove("BACKGROUND_Google")
        labels.sort()

        self.labels = labels

        ### ACCOPPIO LE CLASSE A NUMERI ##
        ### LABELS_INDEX = DICT (NAME : INDEX) ###

        labels_index = {labels[i]: i for i in range(len(labels))}
        #print (labels_index)

        self.labels_index = labels_index
        
        directory = os.path.expanduser("Caltech101/101_ObjectCategories")


        ## CREO COPPIE PATH,LABEL ##
        ## INSTANCES = LIST [ PATH,INDEX ]
        instances = []

        for i in range(0, len(dataset)):
            riconoscimento = dataset[i].split("/")[0]
           
            path = 'Caltech101/101_ObjectCategories/' + dataset[i]
            
            class_index = labels_index.get(riconoscimento)
            item = (path[:len(path)-1],class_index)
            instances.append(item)
            
        self.instances = instances
        
        ## CREO VETTORE LABELS ##

        targets = [s[1] for s in instances]
        self.targets = targets


        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        

        image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int
        '''
        # Applies preprocessing when accessing the image
       if self.transform is not None:
            image = self.transform(image)

        path, target = self.instances[index]
        print(path,target)
        img = pil_loader(index)
        print(img,target)

        return img, target


    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.instances) # Provide a way to get the length (number of elements) of the dataset
        return length
