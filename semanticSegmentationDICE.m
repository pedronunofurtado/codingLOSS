
% get Data
tStartSemSegEach = tic;

imDir = fullfile(baseDataDir,imDirSUFFIX);
if(isUNICLASS)
    imDir = fullfile(baseDataDir(1:end-3),imDirSUFFIX);
end

pxDir = fullfile(baseDataDir,pxDirSUFFIX);

if(~isempty(whichDir))
    pxDir=fullfile(pxDir,whichDir);
end

if(isUNICLASS)
    pxDir = fullfile(baseDataDir,'\a. Training Set')
end

imds = imageDatastore(imDir);
%NAO DA:imds = augmentedImageDatastore(inputSize,imds);


pxds = pixelLabelDatastore(pxDir,classNames,pixelLabelID);

% test data

imDirTEST = fullfile(baseDataDir,imDirTESTSUFFIX);
if(isUNICLASS)
    imDirTEST = fullfile(baseDataDir(1:end-3),imDirTESTSUFFIX);
end

imdsTEST = imageDatastore(imDirTEST);

pxDirTEST = fullfile(baseDataDir,pxDirTESTSUFFIX);

if(~isempty(whichDir))
    pxDirTEST=fullfile(pxDirTEST,whichDir);
end

if(isUNICLASS)
    pxDirTEST = fullfile(baseDataDir,'\b. Testing Set')
end

pxdsTEST = pixelLabelDatastore(pxDirTEST,classNames,pixelLabelID);


% build net
numClasses=length(classNames);
%inputSize=[32 32 3]; 
numFilters=64; 
filterSizeDN=4; 
filterSizeUP=4;
layers = buildSemSegNetOther(whichNet,numClasses, inputSize, numFilters, filterSizeDN, filterSizeUP);

trainingData = pixelLabelImageDatastore(imds,pxds);

% balance last layer:
tbl = countEachLabel(trainingData);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
frequency(frequency==0)=0.01;
classWeights = 1./frequency;


if(~loss.isUseWeights)
    classWeights(:)=1;
end


if(loss.isRemoveBKGND)
    classWeights(1)=0;
end

if(loss.isAdaptive) % used only in iou tversky,
    classWeights=[]; % detected by empty classweights
end


if(strcmp(loss.lossi,'iou11')) alpha=1; beta=1;
elseif(strcmp(loss.lossi,'iou1505')) alpha=1.5; beta=0.5;
elseif(strcmp(loss.lossi,'iou0515')) alpha=0.5; beta=1.5;
end

if(strcmp(loss.lossi,'crossentropy'))
    pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
elseif(strcmp(loss.lossi,'diceWeights'))
    pxLayer = diceWeightsPixelClassificationLayer('diceWeights',classWeights, miniBatchSize);
elseif(strcmp(loss.lossi,'iou11') || strcmp(loss.lossi,'iou1505') || strcmp(loss.lossi,'iou0515'))  
    pxLayer = tverskyPixelClassificationLayer('tversky', alpha, beta,classWeights, miniBatchSize);    
else  
    disp("ERRO: pixelLayer mal definida");
    throw exception;
end

%pxLayer = dicePixelClassificationLayer('dice');
%pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
if(strcmp(whichNet,'SEGNET'))
    layers = replaceLayer(layers,"pixelLabels",pxLayer);
elseif(strcmp(whichNet,'UNET'))
    layers = replaceLayer(layers,"Segmentation-Layer",pxLayer);    
elseif(strcmp(whichNet,'FCN'))
     layers = replaceLayer(layers,"pixelLabels",pxLayer);
elseif(strcmp(whichNet,'DEEPLAB'))
     layers = replaceLayer(layers,"classification",pxLayer);
else    
    layers(end) = pixelClassificationLayer('Classes',tbl.Name,'ClassWeights',classWeights);
end

    imDirTEST = fullfile(baseDataDir,strcat(imDirTESTSUFFIX,'TEST'));
    if(isUNICLASS)
        imDirTEST = fullfile(baseDataDir(1:end-3),strcat(imDirTESTSUFFIX,'TEST'));
    end
    
    imdsTEST = imageDatastore(imDirTEST);  
    
    
    imDirVAL = fullfile(baseDataDir,strcat(imDirTESTSUFFIX,'VAL'));    
    if(isUNICLASS)
        imDirVAL = fullfile(baseDataDir(1:end-3),strcat(imDirTESTSUFFIX,'VAL'));
    end
    
    imdsVAL = imageDatastore(imDirVAL);
    
    pxDirTEST = fullfile(baseDataDir,strcat(pxDirTESTSUFFIX,'TEST'));
    if(~isempty(whichDir))
        pxDirTEST=fullfile(pxDirTEST,strcat(whichDir,'TEST'));
    end
    if(isUNICLASS)
        pxDirTEST = fullfile(baseDataDir,'b. Testing SetTEST');
    end
    
    pxdsTEST = pixelLabelDatastore(pxDirTEST,classNames,pixelLabelID);
    
    pxDirVAL = fullfile(baseDataDir,strcat(pxDirTESTSUFFIX,'VAL'));
    if(~isempty(whichDir))
        pxDirVAL=fullfile(pxDirTEST,strcat(whichDir,'VAL'));
    end
    if(isUNICLASS)
         pxDirVAL = fullfile(baseDataDir,'b. Testing SetVAL');
    end

    pxdsVAL = pixelLabelDatastore(pxDirVAL,classNames,pixelLabelID);
    
    % Define validation data.
    pximdsVal = pixelLabelImageDatastore(imdsVAL,pxdsVAL);    
    
% Define training options. 
learnRateDropPeriod=25;
learnRateDropFactor=0.8;
momentum=0.9;
initialLearnRate=learnrate;
l2Regularization=0.005;
gradientThreshold=10;

options = trainingOptions('sgdm', ...
        'LearnRateSchedule','piecewise',...
        'LearnRateDropPeriod',learnRateDropPeriod,...
        'LearnRateDropFactor',learnRateDropFactor,...
        'Momentum',momentum, ...
        'InitialLearnRate',learnrate, ...
        'L2Regularization',l2Regularization, ...
        'GradientThreshold',gradientThreshold,...
        'ValidationData',pximdsVal,...
        'MaxEpochs',nepochs, ...  
        'MiniBatchSize',miniBatchSize,... 
        'Shuffle','every-epoch', ...
        'VerboseFrequency',10,...
        'Plots','training-progress',...
        'ValidationPatience', Inf); ...      
%    augmenter = imageDataAugmenter('RandXReflection',true,...
%     'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

   augmenter = imageDataAugmenter('RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

    %train dataset
    pximds = pixelLabelImageDatastore(imds,pxds, ...
    'DataAugmentation',augmenter);

    [net, info] = trainNetwork(pximds,layers,options);
     


tElapsedSemSegEach = toc(tStartSemSegEach)

resultsDir='results';
mkdir(resultsDir);

save(strcat(resultsDir,'/','net',whichName),'net');


isSaveResults=true;

evalSemanticSegmentationNetSave(net,classNames,...
                                imdsTEST, pxdsTEST,...
                                whichName,isSaveResults,...
                                resultsDir);


name=strcat(resultsDir,'/viewTestImages_',whichName);
save(name,'testImages');



              