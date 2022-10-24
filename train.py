from PIL import Image
from torchvision import transforms
from glob2 import glob
import numpy as np

class face_dataset():
    def __init__(self, folder_image=None, transforms=None):
        self.folder_image = folder_image
        self.transforms = transforms
        self.path_img = [glob(folder_face) for folder_face in glob(folder_image + '/*')]

    def __getitem__(self, index):
        y = np.random.randint(2, size=1)

        if y == 0:
            # name = np.random.choice(f[i], 2, replace=False)
            index_temp = np.random.randint(len(self.path_img))
            while index_temp == index:
                index_temp = np.random.randint(len(self.path_img))
            path_img1 = np.random.choice(self.path_img[index], 1)
            path_img2 = np.random.choice(self.path_img[index], 1)
            img1 = Image.open(path_img1)
            img2 = Image.open(path_img2)
        else:
            path_img1, path_img2 = np.random.choice(self.path_img[index], 2, replace=False)
            img1 = Image.open(path_img1)
            img2 = Image.open(path_img2)
        
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, y
        
    def __len__(self):
        return len(self.path_img)
