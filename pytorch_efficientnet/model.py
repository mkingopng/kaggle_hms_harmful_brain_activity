"""
asdf
"""
from config import *


# model
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        """
        custom model using pretrained spine and 4 custom layers
        :param config:
        :param num_classes:
        :param pretrained:
        """
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate=CFG.DROP_RATE,
            drop_path_rate=CFG.DROP_PATH_RATE,
        )

        if config.FREEZE:
            for i, (name, param) in enumerate(list(self.model.named_parameters())[0:config.NUM_FROZEN_LAYERS]):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
            # todo: add more custom layers or improve this
        )

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image
        :param x:
        :return:
        """
        # get spectograms
        spectograms = [x[:, :, :, i:i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=1)

        # get EEG spectograms
        eegs = [x[:, :, :, i:i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        # reshape (512, 512, 3)
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectograms

        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        """
        forward pass
        :param x:
        :return:
        """
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x