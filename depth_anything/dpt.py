import argparse

import torch
import torch.nn as nn

from blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F

import cv2
from torchvision.transforms import Compose
from util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # print("Shape of layer_1_rn after reassemble + convs:", layer_1_rn.shape)
        # print("First values of layer_1_rn after reassemble + convs:", layer_1_rn[0, 0, :3, :3])

        # print("Shape of layer_4_rn after reassemble + convs:", layer_4_rn.shape)
        # print("First values of layer_4_rn after reassemble + convs:", layer_4_rn[0, 0, :3, :3])

        print("INSIDE FUSION:")
        
        print("Shape of hidden states before feature fusion:", layer_4_rn.shape)
        print("First values before fusion:", layer_4_rn[0, 0, :3, :3])
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])

        # print("Shape of hidden states after feature fusion:", path_4.shape)
        # print("First values after fusion:", path_4[0, 0, :3, :3])

        print("SECOND FUSION")
        print("size:", layer_2_rn.shape[2:])

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])

        print("Shape of path_3 features after feature fusion:", path_3.shape)
        print("First values after second fusion:", path_3[0, 0, :3, :3])

        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        print("Output after feature fusion:", path_1.shape)
        print("First values after feature fusion:", path_1[0, 0, :3, :3])

        # print("Shape of fused hidden states:")
        # print(path_4.shape)
        # print("First values of first fused hidden state:", path_4[0, 0, :3, :3])
        # print(path_3.shape)
        # print(path_2.shape)
        # print(path_1.shape)
        
        out = self.scratch.output_conv1(path_1)

        print("Shape after output_conv1:", out.shape)
        print("First values after output_conv1:", out[0, 0, :3, :3])

        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        print("Shape after output_conv2:", out.shape)
        print("First values after output_conv2:", out[0, 0, :3, :3])
        
        return out
        
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument(
        "--size",
        default="small",
        type=str,
        choices=["small", "base", "large"],
        help="Name of the model you'd like to convert.",
    )
    args = parser.parse_args()
    size = args.size

    from huggingface_hub import hf_hub_download

    if size == "small":
        depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384]).eval()
        filepath = hf_hub_download(
            repo_id="LiheYoung/Depth-Anything", filename="checkpoints/depth_anything_vits14.pth", repo_type="space"
        )
    elif size == "base":
        depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768]).eval()
        filepath = hf_hub_download(
            repo_id="LiheYoung/Depth-Anything", filename="checkpoints/depth_anything_vitb14.pth", repo_type="space"
        )
    elif size == "large":
        depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).eval()
        filepath = hf_hub_download(
            repo_id="LiheYoung/Depth-Anything", filename="checkpoints/depth_anything_vitl14.pth", repo_type="space"
        )

    state_dict = torch.load(filepath, map_location="cpu")
    depth_anything.load_state_dict(state_dict)

    # load image
    from PIL import Image
    import requests

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # predict depth
    import numpy as np

    original_transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    original_pixel_values = original_transform({'image': np.array(image)})["image"]
    original_pixel_values = torch.from_numpy(original_pixel_values).unsqueeze(0)
    print("Shape of original pixel values:", original_pixel_values.shape)
    print("Mean of original pixel values:", original_pixel_values.mean())

    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize((518, 518), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = transform(image).unsqueeze(0)

    print(image.shape)

    depth = depth_anything(image)

    print(depth.shape)
    