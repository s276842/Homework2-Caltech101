from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
from tqdm import tqdm
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.dataset_path = os.path.join(self.root, "101_ObjectCategories")
        self.categories = os.listdir(self.dataset_path)
        self.categories.remove("BACKGROUND_Google")
        self.data = []

        self.categories_distr = {}
        self.__labels_to_int = {cat:i for i,cat in enumerate(self.categories)}
        self.__int_to_labels = {i:cat for i, cat in enumerate(self.categories)}

        with open(os.path.join(self.root, f"{split}.txt")) as f:
            for line in tqdm(f):
                if "BACKGROUND_Google" in line:
                    continue

                try:
                    line = line.strip()
                    label, img_path = line.split('/')
                    self.data.append([os.path.join(self.dataset_path, line), self.__labels_to_int[label]])
                    self.categories_distr[label] = self.categories_distr.get(label,0) + 1
                except Exception as e:
                    print(str(e))

        np.random.shuffle(self.data)



    def __getitem__(self, index, label_int=False):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        img_path, label = self.data[index]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = self.data.__len__()
        return length

    def __str__(self):
        return f"Loaded {self.split}set ({self.__len__()} entries):\n"+self.categories_distr.__str__()

    def int_to_label(self, val):
        return self.__int_to_labels[val]



if __name__ == '__main__':
    set = Caltech('.', split='train')
    img, target = set[0]

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

    print(f"Label: {set.int_to_label(target)}")
