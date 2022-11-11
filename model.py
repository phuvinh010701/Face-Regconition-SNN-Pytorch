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

class contrastive_loss(nn.CosineEmbeddingLoss):

    def __init__(self, margin: float = 0.4, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(margin)

    def accuracy(self, x0, x1, y):
        cos = nn.CosineSimilarity(dim=1)
        dist = cos(x0, x1)
        pred = torch.Tensor([1 if dist_ < 0.001 else -1 for dist_ in dist]).cuda()
        acc = torch.sum(pred == y) / len(y)
        return acc
