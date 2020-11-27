from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
from tqdm import tqdm
import sys
import numpy as np


IMG_SIZE = 50

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

        self.transform = transform
        self.target_transform = target_transform

        self.dataset_path = os.path.join(self.root, "101_ObjectCategories")
        self.splitfile_path = os.path.join(self.root, f"{split}.txt")
        self.categories = os.listdir(self.dataset_path)
        self.categories.remove("BACKGROUND_Google")
        self.data = []
        self.categories_distr = {}
        self.category_label = {cat:i for i,cat in enumerate(self.categories)}

        with open(self.splitfile_path) as f:
            for line in tqdm(f):
                if "BACKGROUND_Google" in line:
                    continue

                try:
                    label, img_path = line.split('/')
                    line = line.strip()
                    img = pil_loader(os.path.join(self.dataset_path, line))
                    self.data.append([img, label])
                    self.categories_distr[label] = self.categories_distr.get(label,0) + 1
                except Exception as e:
                    print(str(e))

        np.random.shuffle(self.data)

    def labels(self):
        return self.category_label

    def preprocess(self):
        self.data = []
        for img in self.imgs:
            img = img.resize((IMG_SIZE, IMG_SIZE))
            self.data.append(np)


    def __getitem__(self, index, label_int=False):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.data[index] # Provide a way to access image and label via index
                                        # Image should be a PIL Image
                                        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        if label_int:
            return image, self.category_label[label]

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = self.data.__len__()# Provide a way to get the length (number of elements) of the dataset
        return length

    def __str__(self):
        return f"Loaded {self.split}set ({self.__len__()} entries):\n"+self.categories_distr.__str__()

if __name__ == '__main__':
    cal = Caltech(r"C:\Users\Kaloo\Documents\pycharm_projects\MLDL_hw2\Caltech101", split='train')
    print(cal)

    img, lab = cal[0]
    labels_dict = cal.labels()
    print(f"Image 0 class {labels_dict[lab]}")