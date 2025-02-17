import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ..model.clip_utils.utils import get_labelset_from_dataset

class Adapter_Yes(nn.Module):
      # Text-guided Fusion Adapter
    def __init__(self):
            super(Adapter_Yes, self).__init__()


            self.adapter_layer = nn.Sequential(
                        nn.Linear(
                        in_features=512,
                        out_features=768,
                        bias=False),
                        nn.LayerNorm(1024, elementwise_affine=False),
                        nn.GELU())
            # self.adapter_layer = self.adapter_layer.half()
            
            self.classifier = nn.Linear(
                        in_features=768,
                        out_features=512,
                        bias=False)
            # self.classifier = self.classifier.half()


            self.init_weights()

    def forward(self, fea):
            fea = fea.float()
            fea = self.adapter_layer(fea)
            logits = self.classifier(fea)
            
            return logits

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                        init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                        init.constant_(m.weight, 1)
                        init.constant_(m.bias, 0)

class Adapter_No(nn.Module):
      # Text-guided Fusion Adapter
    def __init__(self):
            super(Adapter_No, self).__init__()


            self.adapter_layer = nn.Sequential(
                        nn.Linear(
                        in_features=768,
                        out_features=1024,
                        bias=False),
                        nn.BatchNorm1d(1024, momentum=0.1),
                        nn.GELU())
            # self.adapter_layer = self.adapter_layer.half()
            
            self.classifier = nn.Linear(
                        in_features=1024,
                        out_features=768,
                        bias=False)
            # self.classifier = self.classifier.half()


            self.init_weights()

    def forward(self, fea):
        #     fea = fea.float()
            fea = self.adapter_layer(fea)
            logits = self.classifier(fea)
            
            return logits

    def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                        init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                        init.constant_(m.weight, 1)
                        init.constant_(m.bias, 0)