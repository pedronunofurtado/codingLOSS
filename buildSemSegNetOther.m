function [net] = buildSemSegNetOther(whichNet, numClasses, inputSize, numFilters, filterSizeDN, filterSizeUP) 

if(strcmp(whichNet,'UNET'))
    net = unetLayers(inputSize,numClasses);
elseif(strcmp(whichNet,'SEGNET'))
    encoderDepth = 4;
    net = segnetLayers(inputSize,numClasses,encoderDepth);
elseif(strcmp(whichNet,'FCN'))
    net = fcnLayers(inputSize(1:2),numClasses,'Type','8s');
elseif(strcmp(whichNet,'DEEPLAB'))
    net = helperDeeplabv3PlusResnet18(inputSize, numClasses);
else % 'NORMAL'
    net=buildSemSegNetwork(numClasses, inputSize, numFilters, filterSizeDN, filterSizeUP); 
end

% FCNLAYERS

% '32s'	
% Upsamples the final feature map by a factor of 32. This option provides coarse segmentation with a lower computational cost.
% 
% '16s'	
% Upsamples the final feature map by a factor of 16 after fusing the feature map from the fourth pooling layer. This additional information from earlier layers provides medium-grain segmentation at the cost of additional computation.
% 
% '8s'	
% Upsamples the final feature map by a factor of 8 after fusing feature maps from the third and fourth max pooling layers. This additional information from earlier layers provides finer-grain segmentation at the cost of additional computation.