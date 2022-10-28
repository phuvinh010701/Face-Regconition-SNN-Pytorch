import itertools
import pandas as pd
import glob2
import numpy as np

folder_path = "../Extracted_Faces/*"
folder_imgs = glob2.glob(folder_path)
list_path_img = [glob2.glob(folder_img + "/*") for folder_img in folder_imgs]

imgs1 = []
imgs2 = []
labels = []

### Create label 1
for i in range(len(list_path_img)):

    count_subset = 0

    for subset in itertools.combinations(list_path_img[i], 2):
        
        imgs1.append(subset[0][2:])
        imgs2.append(subset[1][2:])
        labels.append(1)

        count_subset += 1
        if count_subset >= 3:
            break

### Create label 0
balance_data = len(imgs1)
for i in range(balance_data):
    idx1, idx2 = np.random.randint(len(list_path_img)), np.random.randint(len(list_path_img))
    if idx1 != idx2:
        path1 = np.random.choice(list_path_img[idx1], 1)[0][2:]
        path2 = np.random.choice(list_path_img[idx2], 1)[0][2:]
        imgs1.append(path1)
        imgs2.append(path2)
        labels.append(0)
    

dict_ = {"img1": imgs1, "img2": imgs2, "label": labels}
df = pd.DataFrame(dict_)
 
df.to_csv('../data_path.csv', header=True, index=False)