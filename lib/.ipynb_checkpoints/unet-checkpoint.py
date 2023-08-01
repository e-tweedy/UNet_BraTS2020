import torch
from torch import nn
from torchvision.transforms.functional import center_crop

def conv_layer(dim:int):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d

def conv_trans_layer(dim:int):
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d

def max_pool_layer(dim:int):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d

def group_norm_layer():
    return nn.GroupNorm
    
def batch_norm_layer(dim:int):
    if dim == 3:
        return nn.BatchNorm3d
    elif dim == 2:
        return nn.BatchNorm2d

class DoubleConv(nn.Module):
    """
    Block consisting of a composition of two copies
    of the following sequence:
    Conv -> GroupNorm -> ReLU
    Parameters:
    -----------
    in_channels,out_channels : int,int
        in_channels is number of input channels in first conv
        out_channels is number of output_channels of first conv
        and input and output channels of second conv
    dim : int
        The desired dimension - must be 2 or 3
        Use 2 if inputs will have shape (batch_size,channel_num,H,W)
        and use 3 if (batch_size,channel_num,H,W,D)
    num_groups : int
        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                dim = 2,
                num_groups = 4,
                ):
        super().__init__()

        self.conv = nn.Sequential(
            conv_layer(dim)(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm_layer()(num_channels = out_channels, num_groups = num_groups),
            nn.ReLU(inplace=True),
            conv_layer(dim)(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm_layer()(num_channels = out_channels, num_groups = num_groups),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet module class for 2-dimensional
    or 3-dimensional images
    Parameters:
    -----------
    in_channels, out_channels : int
        The numbers of input and output channels
        out_channels should be number of segmentation
        label classes
    dim : int
        The desired dimension - must be 2 or 3
        Use 2 if inputs will have shape (batch_size,channel_num,H,W)
        and use 3 if (batch_size,channel_num,H,W,D)
    init_features : int
        The number of output features in the first
        encoder convolution block.
        - Each successive encoder convolution block
          has twice as many features.
        - Each encoder convolution block is mirrored
          by a decoder up-convolution block of with
          the same number of output features
    num_stages : int
        The number of convolution blocks in the encoder.
        The decoder will have the same number of
        up-convolution blocks.
    """
    def __init__(
        self, in_channels = 3, out_channels = 3, dim = 2, init_features = 64, num_stages = 4,
    ):
        super().__init__()
        # Set up feature counts for encoder/decoder blocks
        self.features = [init_features*2**k for k in range(num_stages)]
        # Set up layer lists
        self.enc_layers = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        self.pool = max_pool_layer(dim)(kernel_size = 2, stride = 2)

        # Loop through feature list constructing encoder blocks
        # Pooling will be incorporated into forward method
        for feature in self.features:
            self.enc_layers.append(DoubleConv(in_channels,feature,dim))
            in_channels = feature

        # Loop backwards through feature list constructing decoder blocks
        for feature in reversed(self.features):
            self.dec_layers.append(
                conv_trans_layer(dim)(2*feature, feature, kernel_size = 2, stride = 2)
            )
            self.dec_layers.append(DoubleConv(2*feature, feature, dim))
        # One last convolution block at the bottom of the "U"
        self.bottleneck = DoubleConv(self.features[-1],2*self.features[-1],dim)
        # Final convolution on the output side
        self.final_conv = conv_layer(dim)(self.features[0],out_channels,kernel_size = 1)

    def forward(self,x):
        skip_connections = []

        # Max-pooling map after each encoder convolution block,
        # but also set aside pre-images of max-pooling maps
        # for skip connections
        for layer in self.enc_layers:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # Skip connection inputs are used in reverse by the decoder
        skip_connections = skip_connections[::-1]

        # Note len(self.dec_layers) == 2*len(self.enc_layers) == 2*len(skip_connections)
        # Every other decoder layer is a convolution block,
        # and we loop through their inputs concatenating skip
        # connection elements onto them
        for idx in range(0,len(self.dec_layers),2):
            x = self.dec_layers[idx](x)
            skip_connection = skip_connections[idx//2]

            # Resize before skip_connection before concatenation if necessary
            if x.shape != skip_connection.shape:
                skip_connection = center_crop(skip_connection,x.shape[2])

            concat_skip = torch.cat((skip_connection,x),dim=1)
            # Pass through the up-convolution layer that follows
            x = self.dec_layers[idx+1](concat_skip)

        return self.final_conv(x)