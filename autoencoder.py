import torchvision.models as models
import pytorch_lightning as pl
from torch import nn


class ImagenetTransferAutoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # init a pretrained resnet
        num_target_classes = 10
        #, avg_pool="AdaptiveAvgPool2d"
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        self.classifier = nn.Linear(1000, num_target_classes)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x 