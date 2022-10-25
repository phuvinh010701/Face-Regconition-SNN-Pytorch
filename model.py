from torch import nn
import torch

class siamese(nn.Module):
    def __init__(self):
        super(siamese, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        self.fc = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
        )
        
    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size(), -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class contrastive_loss(nn.Module):

    def __init__(self, margin=2.0):
        super(contrastive_loss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        
        dist = torch.cdist(x0, x1)
        max_= torch.clamp(self.margin - dist, min=0.0)
        loss = (1 - y) * pow(dist, 2) + y * torch.pow(max_, 2)
        # diff = x0 - x1
        # dist_sq = torch.sum(torch.pow(diff, 2), 1)
        # dist = torch.sqrt(dist_sq)

        # mdist = self.margin - dist
        # dist = torch.clamp(mdist, min=0.0)
        # loss = y * torch.pow(dist_sq, 2) + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()
        return loss