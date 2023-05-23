import torchvision.models as models
from torch import nn
from torchvision.models import ResNet50_Weights


class ResNet:
    def __init__(self, pretrained=False, num_classes=None):
        net = models.resnet50(pretrained=pretrained)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features=in_features, out_features=num_classes)
        self.model = net

    def get_model(self):
        return self.model
