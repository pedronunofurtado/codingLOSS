Pedro Furtado@UCoimbra

This is preliminary code dump that I will be structuring and improving to become presentable.

0. calling_MRI_CT_IDRID.m

specification of properties to run experiments

1. semanticSegmentationDICE.m: code preparing and training the networks with any defined loss function:

example of training options:

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

2. Piece that builds networks:

buildSemSegNetOther.m

3. Loss, by replacing pixel classification layers:

diceWeightsPixelClassificationLayer
dicePixelClassificationLayer
tverskyPixelClassificationLayer

4. Evaluating the approaches using independent test dataset:

evalSemanticSegmentationNetSave

5. Testing varied loss functions
runLossExperiments.m
runLosses.m
