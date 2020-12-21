function [net] = buildSemSegNetwork(numClasses, inputSize, numFilters, filterSizeDN, filterSizeUP) 

% Create An Image Input Layer

% A semantic segmentation network starts with an imageInputLayer, which defines the smallest image size the network can process. Most semantic segmentation networks are fully convolutional, which means they can process images that are larger than the specified input size. 
% Here, an image size of [32 32 3] is used for the network to process 64x64 RGB images.


%VAR inputSize = [32 32 3];
imgLayer = imageInputLayer(inputSize)


% Create Downsampling Network

% Start with the convolution and ReLU layers. The convolution layer padding is selected such that the output size of the convolution layer is the same as the input size. This makes it easier to construct a network because the input and output sizes between most layers remain the same as you progress through the network.


%VAR filterSize = 3;
%VAR numFilters = 32;
conv = convolution2dLayer(filterSizeDN,numFilters,'Padding',1);
relu = reluLayer();

% The downsampling is performed using a max pooling layer. Create a max
%pooling layer to downsample the input by a factor of 2 by setting the 
%'Stride' parameter to 2.

poolSize = 2;
maxPoolDownsample2x = maxPooling2dLayer(poolSize,'Stride',2,'Padding',1);

% Stack the convolution, ReLU, and max pooling layers to create a network that downsamples its input by a factor of 4.

downsamplingLayers = [
    conv
    relu
    maxPoolDownsample2x
    conv
    relu
    maxPoolDownsample2x
    conv
    relu
    maxPoolDownsample2x    
    ]

% Create Upsampling Network

% The upsampling is done using the tranposed convolution layer (also commonly referred to as "deconv" or "deconvolution" layer). When a transposed convolution is used for upsampling, it performs the upsampling and the filtering at the same time.

% Create a transposed convolution layer to upsample by 2.

%VAR filterSize = 4;
transposedConvUpsample2x = transposedConv2dLayer(filterSizeUP,numFilters,'Stride',2,'Cropping',1);

% The 'Cropping' parameter is set to 1 to make the output size equal twice the input size.

% Stack the transposed convolution and relu layers. An input to this set of layers is upsampled by 4.

upsamplingLayers = [
    transposedConvUpsample2x
    relu
    transposedConvUpsample2x
    relu
    transposedConvUpsample2x
    relu    
    ]

% Create A Pixel Classification Layer

% The final set of layers are responsible for making pixel classifications. These final layers process an input that has the same spatial dimensions (height and width) as the input image. However, the number of channels (third dimension) is larger and is equal to number of filters in the last transposed convolution layer. This third dimension needs to be squeezed down to the number of classes we wish to segment. This can be done using a 1-by-1 convolution layer whose number of filters equal the number of classes, e.g. 3.

% Create a convolution layer to combine the third dimension of the input feature maps down to the number of classes.

%VAR numClasses = 3;
conv1x1 = convolution2dLayer(1,numClasses);

% Following this 1-by-1 convolution layer are the softmax and pixel classification layers. These two layers combine to predict the categorical label for each image pixel.

finalLayers = [
    conv1x1
    softmaxLayer()
    pixelClassificationLayer()
    ]

% Stack All Layers

% Stack all the layers to complete the semantic segmentation network.

net = [
    imgLayer    
    downsamplingLayers
    upsamplingLayers
    finalLayers
    ]

% This network is ready to be trained using trainNetwork from Deep Learning Toolbox?.


end