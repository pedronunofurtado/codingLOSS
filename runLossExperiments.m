function []=runLossExperiments(optionsWB,isDoAdaptive,losses,whichNet,nEpochs, lRate, context,...
                    pixelLabelID,baseDataDir,inputSize,...
                    classNames)

for j=1:length(optionsWB)
    loss=struct;
    loss.isInvertWeights=false;
    loss.isUseWeights=optionsWB{j}{1};
    loss.isRemoveBKGND=optionsWB{j}{2};
    loss.isAdaptive=false;

    runLosses(losses,whichNet,nEpochs, lRate, loss,context,pixelLabelID,baseDataDir,inputSize,...
                    classNames);
                %runLosses(losses,whichNet,nEpochs, lRate, loss,context,pixelLabelID,baseDataDir,inputSize,...
                %    classNames) 
end

% if(isDoAdaptive)
% 
% loss=struct;
% loss.isInvertWeights=false;
% loss.isUseWeights=false;
% loss.isRemoveBKGND=false;
% loss.isAdaptive=true;
% 
% runLosses(losses,whichNet,nEpochs, lRate, loss,context,pixelLabelID,baseDataDir,inputSize,...
%                     classNames);
% end

end