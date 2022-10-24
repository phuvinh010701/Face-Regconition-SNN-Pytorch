from PIL import Image
from torchvision import transforms
from glob2 import glob
import numpy as np
from model import *

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



net = siamese().cuda()
criterion = contrastive_loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
siamese_dataset = siamese(raining_dir,transform=transforms.Compose([transforms.Resize((105,105)), transforms.RandomAffine(15),transforms.ToTensor()]))

train_dataloader = torch.utils.data.DataLoader(
    siamese_dataset, shuffle=True, num_workers=8, batch_size=
)


def train():
    loss=[] 
    counter=[]
    iteration_number = 0
    for epoch in range(1,):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()    
        print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
 
    return net
#set the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = train()
torch.save(model.state_dict(), "output/model.pt")
print("Model Saved Successfully") 


def parser():
    parser = argparse.ArgumentParser(description="Unet semantic segmantation")
    parser.add_argument("--img_path", type=str, 
                        help="path to folder contain images")
    parser.add_argument("--label_path", type=str, 
                    help="path to folder contain label")                    
    parser.add_argument("--shape", type=int, default=256)
    parser.add_argument("--batch_size", default=16, type=int,
                        help="number of images to be processed at the same time")
    return parser.parse_args()