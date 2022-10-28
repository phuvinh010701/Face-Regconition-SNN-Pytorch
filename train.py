from PIL import Image
from torchvision import transforms
from model import *
import pandas as pd
import argparse

class face_dataset():
    def __init__(self, data_path=None, folder_path=None, transforms=None):
        
        self.transforms = transforms
        self.data_path = pd.read_csv(data_path, header=False)
        self.folder_path = folder_path
        
    def __getitem__(self, index):
        
        img1_path, img2_path, y = self.data_path[index]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
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
    parser.add_argument("--data_path", type=str, help="number of epochs")
    parser.add_argument("--folder_path", type=str, help="path to folder contain images")                  
    parser.add_argument("--batch_size", default=16, type=int, help="number of images to be processed at the same time")
    parser.add_argument("--epochs", default=20, type=int, help="number of epochs")
    return parser.parse_args()

def main():
    args = parser()
    net = siamese().cuda()
    criterion = contrastive_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_dataset = face_dataset(
        args.folder_path,
        transforms=transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor()]),
    )

    train_dataloader = torch.utils.data.DataLoader(
        siamese_dataset, shuffle=True, num_workers=8, batch_size=args.batch_size
    )
    model = train(net, train_dataloader, optimizer, criterion, args.epochs)
    torch.save(model.state_dict(), "output/model.pt")
    print("Model Saved Successfully") 

if __name__ == "__main__":
    main()