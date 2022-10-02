import timm
from torch import nn


class FeatureExtractor(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.feature_extractor = c.feature_extractor
        self.img_size = c.img_size[0]

        self.fe_model = timm.create_model(self.feature_extractor, pretrained=True)

    def forward(self, x):
        x = self.fe_model.patch_embed(x)
        x = x + self.fe_model.pos_embed
        x = self.fe_model.pos_drop(x)

        for i in range(41):  # paper Table 6. Block Index = 40
            x = self.fe_model.blocks[i](x)

        N, _, C = x.shape
        x = self.fe_model.norm(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(N, C, self.img_size // 16, self.img_size // 16)

        features = x

        return features
