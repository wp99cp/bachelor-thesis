# import the necessary packages
import torch
import torch.nn.functional as functional
from torch.nn import Conv2d, Sequential, BatchNorm2d
from torch.nn import ConvTranspose2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop

from configs.config import IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES


# Next, we define a Block module as the building unit of our encoder and decoder architecture.
# It is worth noting that all models or model sub-parts that we define are required to inherit
# from the PyTorch Module class, which is the parent class in PyTorch for all neural network modules.


class DoubleConv(Module):
    """
    source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py#L35
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, inChannels, outChannels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = outChannels

        self.double_conv = Sequential(
            Conv2d(inChannels, mid_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(mid_channels),
            ReLU(inplace=True),
            Conv2d(mid_channels, outChannels, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(outChannels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(Module):

    def __init__(self, inChannels, outChannels):
        super().__init__()

        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, 3)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class DownBlock(Module):

    def __init__(self, inChannels, outChannels):
        super().__init__()

        # store the convolution and RELU layers
        self.maxpool2d = MaxPool2d(2)
        self.doubleConv = DoubleConv(inChannels, outChannels)

    def forward(self, x):
        # apply MaxPool => DoubleConv block to the inputs and return it
        return self.doubleConv(self.maxpool2d(x))


class Encoder(Module):
    """

    The encoder module is responsible for extracting the features from the input image.
    It consists of a series of convolutional blocks, each followed by a max pooling layer.

    """

    def __init__(self, channels):
        """
        Initialize the encoder module.
        :param channels: a tuple containing the number of channels in each encoder block
        """

        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList([
            DownBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

        self.pool = MaxPool2d(2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []

        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)

        # return the list containing the intermediate outputs
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels):
        super().__init__()

        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList([
            ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2)
            for i in range(len(channels) - 1)
        ])

        self.dec_blocks = ModuleList([
            UpBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)

            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)

        # return the final decoder output
        return x

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)

        # return the cropped features
        return encFeatures


class UNet(Module):

    def __init__(self, encChannels=(NUM_CHANNELS, 64, 128, 256),
                 decChannels=(256, 128, 64),
                 nbClasses=NUM_CLASSES, retainDim=True,
                 outSize=(IMAGE_SIZE, IMAGE_SIZE)):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, kernel_size=1)
        self.retainDim = retainDim
        self.outSize = outSize

        # map

    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)

        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])

        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)

        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = functional.interpolate(map, self.outSize)

        # return the segmentation map
        return torch.softmax(map, dim=1)
