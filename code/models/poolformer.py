import torch
from torch import nn, Tensor
from libs.common import DropPath
from libs.common import CBAM

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class PatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, padding)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        return x


class Pooling(nn.Module):
    def __init__(self, pool_size=3) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size//2, count_include_pad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x) - x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        # self.act = nn.GELU()
        self.act = StarReLU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, dpr=0., layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.mlp = MLP(dim, int(dim*4))
        
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))) 
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))) 
        return x

poolformer_settings = {
    'S24': [[4, 4, 12, 4], [64, 128, 320, 512], 0.1],       # [layers, embed_dims, drop_path_rate]
    'S36': [[6, 6, 18, 6], [64, 128, 320, 512], 0.2],
    'M36': [[6, 6, 18, 6], [96, 192, 384, 768], 0.3]
}


class PoolFormer(nn.Module):     
    def __init__(self, model_name: str = 'S24') -> None:
        super().__init__()
        assert model_name in poolformer_settings.keys(), f"PoolFormer model name should be in {list(poolformer_settings.keys())}"
        layers, embed_dims, drop_path_rate = poolformer_settings[model_name]
        self.channels = embed_dims
    
        self.patch_embed = PatchEmbed(7, 4, 2, 3, embed_dims[0])

        network = []

        for i in range(len(layers)):
            blocks = []
            for j in range(layers[i]):
                dpr = drop_path_rate * (j + sum(layers[:i])) / (sum(layers) - 1)
                blocks.append(PoolFormerBlock(embed_dims[i], 3, dpr))

            network.append(nn.Sequential(*blocks))
            if i >= len(layers) - 1: break
            network.append(PatchEmbed(3, 2, 1, embed_dims[i], embed_dims[i+1]))

        self.network = nn.ModuleList(network)

        self.out_indices = [0, 2, 4, 6]
        for i, index in enumerate(self.out_indices):
            self.add_module(f"norm{index}", nn.GroupNorm(1, embed_dims[i]))

    def forward(self, x: Tensor):
        x = self.patch_embed(x)
        outs = []

        for i, blk in enumerate(self.network):
            x = blk(x)

            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(x)
                outs.append(out)
        return outs
    
class ImageEncoder(nn.Module):

    def __init__(self, encode_size=14, embed_dim=512):
        """
        param:
        encode_size:    encoded image size.
                        int

        embed_dim:      encoded images features dimension
                        int
        """
        super(ImageEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.encode_size = encode_size
        # pretrained ImageNet ResNet-101
        # Remove last linear and pool layers
        self.backbone = PoolFormer('S24')
        self.backbone.load_state_dict(torch.load('/home/ai/lab208/image-cationing/image_captioning_with_transformers/code/checkpoints/poolformer_s24.pth', map_location='cpu'), strict=False)

        self.downsampling = nn.Conv2d(in_channels=2048,
                                      out_channels=embed_dim,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

        # Resize images, use 2D adaptive max pooling
        self.adaptive_resize = nn.AdaptiveAvgPool2d(encode_size)

    def forward(self, images: Tensor):
        """
        param:
        images: Input images.
                Tensor [batch_size, 3, h, w]

        output: encoded images.
                Tensor [batch_size, encode_size * encode_size, embed_dim]
        """
        # batch_size = B
        # image_size = [B, 3, h, w]
        B = images.size()[0]

        # [B, 3, h, w] -> [B, 2048, h/32=8, w/32=8]
        out = self.backbone(images)  # type: Tensor

        # Downsampling: resnet features size (2048) -> embed_size (512)
        # [B, 2048, 8, 8] -> [B, embed_size=512, 8, 8]
        out = self.relu(self.bn(self.downsampling(out)))

        # Adaptive image resize: resnet output size (8,8) -> encode_size (14,14)
        #   [B, embed_size=512, 8, 8] ->
        #       [B, embed_size=512, encode_size=14, encode_size=14] ->
        #           [B, 512, 196] -> [B, 196, 512]
        out = self.adaptive_resize(out)
        out = out.view(B, self.embed_dim, -1).permute(0, 2, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the tuning for blocks 2 through 4.
        """

        for p in self.backbone.parameters():
            p.requires_grad = False

        for c in list(self.backbone.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
