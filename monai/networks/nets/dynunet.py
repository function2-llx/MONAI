# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Sequence
from typing import Literal, Optional, Type, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock, UNetClsOutBlock

UNetBlock = Union[UnetBasicBlock, UnetResBlock]
PoolType = Type[Union[nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]]

__all__ = ["DynUNet", "DynUnet", "Dynunet"]


class DynUNetSkipLayer(nn.Module):
    """
    Defines a layer in the UNet topology which combines the downsample and upsample pathways with the skip connection.
    The member `next_layer` may refer to instances of this class or the final bottleneck layer at the bottom the UNet
    structure. The purpose of using a recursive class like this is to get around the TorchScript restrictions on
    looping over lists of layers and accumulating lists of output tensors which must be indexed. The `heads` list is
    shared amongst all the instances of this class and is used to store the output from the supervision heads during
    forward passes of the network.
    """

    heads: list[torch.Tensor]

    def __init__(self, index, downsample, upsample, next_layer, heads: list[torch.Tensor], super_head=None):
        super().__init__()
        self.downsample = downsample
        self.next_layer = next_layer
        self.upsample = upsample
        self.super_head = super_head
        self.heads = heads
        self.index = index

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        down_out = self.downsample(x)
        if isinstance(self.next_layer, DynUNetSkipLayer):
            bottleneck_out, next_out = self.next_layer(down_out)
        else:
            bottleneck_out = next_out = self.next_layer(down_out)
        up_out = self.upsample(next_out, down_out)
        if self.super_head is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(up_out)

        return bottleneck_out, up_out


class DynUNet(nn.Module):
    """
    This reimplementation of a dynamic UNet (DynUNet) is based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.
    `Optimized U-Net for Brain Tumor Segmentation <https://arxiv.org/pdf/2110.03352.pdf>`_.

    This model is more flexible compared with ``monai.networks.nets.UNet`` in three
    places:

        - Residual connection is supported in conv blocks.
        - Anisotropic kernel sizes and strides can be used in each layers.
        - Deep supervision heads can be added.

    The model supports 2D or 3D inputs and is consisted with four kinds of blocks:
    one input block, `n` downsample blocks, one bottleneck and `n+1` upsample blocks. Where, `n>0`.
    The first and last kernel and stride values of the input sequences are used for input block and
    bottleneck respectively, and the rest value(s) are used for downsample and upsample blocks.
    Therefore, pleasure ensure that the length of input sequences (``kernel_size`` and ``strides``)
    is no less than 3 in order to have at least one downsample and upsample blocks.

    To meet the requirements of the structure, the input size for each spatial dimension should be divisible
    by `2 * the product of all strides in the corresponding dimension`. The output size for each spatial dimension
    equals to the input size of the corresponding dimension divided by the stride in strides[0].
    For example, if `strides=((1, 2, 4), 2, 1, 1)`, the minimal spatial size of the input is `(8, 16, 32)`, and
    the spatial size of the output is `(8, 8, 8)`.

    For backwards compatibility with old weights, please set `strict=False` when calling `load_state_dict`.

    Usage example with medical segmentation decathlon dataset is available at:
    https://github.com/Project-MONAI/tutorials/tree/master/modules/dynunet_pipeline.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        seg_out_channels: number of output channels.
        kernel_size: convolution kernel size.
        strides: convolution strides for each blocks.
        # upsample_kernel_size: convolution kernel size for transposed convolution layers. The values should
        #     equal to strides[1:].
        filters: number of output channels for each blocks. Different from nnU-Net, in this implementation we add
            this argument to make the network more flexible. As shown in the third reference, one way to determine
            this argument is like:
            ``[64, 96, 128, 192, 256, 384, 512, 768, 1024][: len(strides)]``.
            The above way is used in the network that wins task 1 in the BraTS21 Challenge.
            If not specified, the way which nnUNet used will be employed. Defaults to ``None``.
        dropout: dropout ratio. Defaults to no dropout.
        norm_name: feature normalization type and arguments. Defaults to ``INSTANCE``.
        act_name: activation layer type and arguments. Defaults to ``leakyrelu``.
        # deep_supervision: whether to add deep supervision head before output. Defaults to ``False``.
        #     If ``True``, in training mode, the forward function will output not only the final feature map
        #     (from `output_block`), but also the feature maps that come from the intermediate up sample layers.
        #     In order to unify the return type (the restriction of TorchScript), all intermediate
        #     feature maps are interpolated into the same size as the final feature map and stacked together
        #     (with a new dimension in the first axis)into one single tensor.
        #     For instance, if there are two intermediate feature maps with shapes: (1, 2, 16, 12) and
        #     (1, 2, 8, 6), and the final feature map has the shape (1, 2, 32, 24), then all intermediate feature maps
        #     will be interpolated into (1, 2, 32, 24), and the stacked tensor will has the shape (1, 3, 2, 32, 24).
        #     When calculating the loss, you can use torch.unbind to get all feature maps can compute the loss
        #     one by one with the ground truth, then do a weighted average for all losses to achieve the final loss.
        deep_supr_num: number of feature maps that will output during deep supervision head. The
            value should be larger than 0 and less than the number of up sample layers.
            Defaults to 1.
        res_block: whether to use residual connection based convolution blocks during the network.
            Defaults to ``False``.
        trans_bias: whether to set the bias parameter in transposed convolution layers. Defaults to ``False``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        cls_out_channels: int,
        seg_out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        # upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        output_paddings: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[tuple, str, float]] = None,
        norm_name: Union[tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
        pool_type: str = 'max',
        pool_fmap: int = 2,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.cls_out_channels = cls_out_channels
        self.seg_out_channels = seg_out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_paddings = output_paddings
        # self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.conv_block: Type[UNetBlock] = UnetResBlock if res_block else UnetBasicBlock
        self.trans_bias = trans_bias
        # self.filters = filters
        if filters is None:
            self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        else:
            self.filters = filters
        self.check_filters()
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.pool_fmap = pool_fmap
        self.pool_type = pool_type
        self.cls_output_block = self.get_cls_output_block(-1)
        self.seg_output_block = self.get_seg_output_block(0)
        self.deep_supr_num = deep_supr_num

        # initialize the typed list of supervision head outputs so that TorchScript can recognize what's going on
        self.heads: list[torch.Tensor] = [torch.empty(1)] * self.deep_supr_num
        self.deep_supervision_heads = self.get_deep_supervision_heads()

        self.apply(self.initialize_weights)
        self.check_kernel_stride()

        def create_skips(
            index: int,
            downsamples: list[nn.Module],
            upsamples: list[nn.Module],
            bottleneck: UNetBlock,
            super_heads: nn.ModuleList,
        ) -> Union[DynUNetSkipLayer, UNetBlock]:
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """

            if len(downsamples) != len(upsamples):
                raise ValueError(f"{len(downsamples)} != {len(upsamples)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                return bottleneck

            # the supervision head used by current layer
            cur_super_head: Optional[UnetOutBlock] = None
            if index > 0 and len(super_heads) > 0:
                cur_super_head, super_heads = super_heads[0], super_heads[1:]

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck, super_heads=super_heads)
            return DynUNetSkipLayer(
                index,
                downsample=downsamples[0],
                upsample=upsamples[0],
                next_layer=next_layer,
                heads=self.heads,
                super_head=cur_super_head,
            )

        self.skip_layers = create_skips(
            index=0,
            downsamples=[self.input_block] + list(self.downsamples),
            upsamples=list(self.upsamples[::-1]),
            bottleneck=self.bottleneck,
            super_heads=self.deep_supervision_heads,
        )

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if len(kernels) != len(strides) or len(kernels) < 3:
            raise ValueError(error_msg)

        for idx, k_i in enumerate(kernels):
            kernel, stride = k_i, strides[idx]
            if not isinstance(kernel, int):
                error_msg = f"length of kernel_size in block {idx} should be the same as spatial_dims."
                if len(kernel) != self.spatial_dims:
                    raise ValueError(error_msg)
            if not isinstance(stride, int):
                error_msg = f"length of stride in block {idx} should be the same as spatial_dims."
                if len(stride) != self.spatial_dims:
                    raise ValueError(error_msg)

    def check_filters(self):
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("length of filters should be no less than the length of strides.")
        else:
            self.filters = filters[: len(self.strides)]

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck_out, up_out = self.skip_layers(x)
        cls_out = self.cls_output_block(bottleneck_out)
        seg_out = self.seg_output_block(up_out)
        if self.training:
            out_all = [seg_out]
            for feature_map in self.heads:
                out_all.append(interpolate(feature_map, seg_out.shape[2:]))
            seg_out = torch.stack(out_all, dim=1)
        return cls_out, seg_out

    def get_input_block(self) -> UNetBlock:
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_bottleneck(self) -> UNetBlock:
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_cls_output_block(self, idx: int) -> UNetClsOutBlock:
        return UNetClsOutBlock(
            self.spatial_dims,
            self.filters[idx],
            self.cls_out_channels,
            self.pool_type,
            self.pool_fmap,
        )

    def get_seg_output_block(self, idx: int) -> UnetOutBlock:
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.seg_out_channels, dropout=self.dropout)

    def get_downsamples(self) -> nn.ModuleList:
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self) -> nn.ModuleList:
        in_channels, out_channels = self.filters[::-1][:-1], self.filters[::-1][1:]
        strides, kernel_size = self.strides[::-1][:-1], self.kernel_size[::-1][:-1]
        output_paddings = self.output_paddings[::-1]
        return self.get_module_list(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            conv_block=UnetUpBlock,
            upsample=True,
            output_paddings=output_paddings,
            trans_bias=self.trans_bias,
        )

    def get_module_list(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: Type[Union[UNetBlock, UnetUpBlock]],
        upsample: bool = False,
        output_paddings: Optional[Sequence[Union[Sequence[int], int]]] = None,
        trans_bias: bool = False,
    ) -> nn.ModuleList:
        layers = []
        if upsample:
            for in_c, out_c, kernel, stride, output_padding in zip(
                in_channels, out_channels, kernel_size, strides, output_paddings
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    'output_padding': output_padding,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)

        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self) -> nn.ModuleList:
        num_up_layers = len(self.strides) - 1
        if self.deep_supr_num >= num_up_layers:
            raise ValueError("deep_supr_num should be less than the number of up sample layers.")
        return nn.ModuleList([self.get_seg_output_block(i + 1) for i in range(self.deep_supr_num)])

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


DynUnet = Dynunet = DynUNet
