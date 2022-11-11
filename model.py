from torch import nn
import torch
from torchvision import models
class siamese(nn.Module):
    def __init__(self):
        super(siamese, self).__init__()
        
        self.base_model = models.resnet101(weights=False)
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model.fc = nn.Sequential(
            nn.Linear(2048, 128)
        )
        
    def forward_once(self, x):
        output = self.base_model(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class contrastive_loss(nn.Module):
    def __init__(self):
        super(contrastive_loss, self).__init__()

    def forward(self, x0, x1, y):

        dist_sq = ((x0 - x1)**2).sum(axis=1)

        dist = torch.sqrt(dist_sq)
        pred = torch.Tensor([1 if dist_ < 0.05 else 0 for dist_ in dist]).cuda()
        acc = torch.mean(torch.sum(pred == y))
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)

        loss = y*dist_sq + (1-y)*torch.pow(dist, 2)
        loss = torch.mean(loss)

        return loss, acc



