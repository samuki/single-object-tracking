import torchvision.models as models
import pytorch_lightning as pl
from torch import nn
import torch

class ImagenetTransferAutoencoder(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # init a pretrained resnet
        num_target_classes = num_classes
        #, avg_pool="AdaptiveAvgPool2d"
        backbone = models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        #x = self.classifier(representations)
        #return x 
        return representations
