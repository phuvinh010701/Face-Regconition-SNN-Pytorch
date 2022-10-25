from PIL import Image
from torchvision import transforms
from glob2 import glob
import numpy as np
from model import *
import argparse
class face_dataset():
    def __init__(self, folder_image=None, transforms=None):
        self.folder_image = folder_image
        self.transforms = transforms
        self.path_img = [glob(folder_face + '/*') for folder_face in glob(folder_image + '/*')]

        # print(self.path_img)
    def __getitem__(self, index):
        y = np.random.randint(2, size=1)[0]

        if y == 1:
            # name = np.random.choice(f[i], 2, replace=False)
            index_temp = np.random.randint(len(self.path_img))
            while index_temp == index:
                index_temp = np.random.randint(len(self.path_img))
            path_img1 = np.random.choice(self.path_img[index], 1)[0]
            path_img2 = np.random.choice(self.path_img[index_temp], 1)[0]
            # print(path_img1, path_img2)
            img1 = Image.open(path_img1[2:])
            img2 = Image.open(path_img2[2:])
        else:
            path_img1, path_img2 = np.random.choice(self.path_img[index], 2, replace=False)
            img1 = Image.open(path_img1[2:])
            img2 = Image.open(path_img2[2:])
        
        img1 = img1.convert("L")
        img2 = img2.convert("L")
        # print(path_img1, path_img2, y)
        # print(img2)
        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        return img1, img2, y

    def __len__(self):
        return len(self.path_img)

def train(model, train_dataloader, optimizer, criterion, epochs):
    loss=[] 
    counter=[]
    iteration_number = 0
    for epoch in range(1, epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = model(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            # print()    
        print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
 
    return model

def parser():
    parser = argparse.ArgumentParser(description="SNN with Pytorch")
    parser.add_argument("--folder_path", type=str, 
                        help="path to folder contain images")                  
    parser.add_argument("--batch_size", default=16, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--epochs", default=20, type=int,
                    help="number of epochs")
    return parser.parse_args()

def main():
    args = parser()
    net = siamese().cuda()
    criterion = contrastive_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_dataset = face_dataset(
        args.folder_path,
        transforms=transforms.Compose([transforms.Resize((105,105)), transforms.RandomAffine(15),transforms.ToTensor()]),
    )

    train_dataloader = torch.utils.data.DataLoader(
        siamese_dataset, shuffle=True, num_workers=8, batch_size=args.batch_size
    )
    model = train(net, train_dataloader, optimizer, criterion, args.epochs)
    torch.save(model.state_dict(), "output/model.pt")
    print("Model Saved Successfully") 

if __name__ == "__main__":
    main()