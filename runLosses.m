
%loss alternatives:
%'crossentropy'
%'diceWeights'
%'iou11' , 'iou1505'  , 'iou0515' (FP alpha)
%'diceLocal'
%'diceIDRID', 'diceCHAOSMRI' , 'diceCHAOSCT'    
%'diceUNIMIB'        
%none of those=default=crossentropy               

for i=1:length(losses)
    if(loss.isAdaptive==true && ~startsWith(losses{i},'iou'))
        continue;
    end
    lossi=losses{i};


    expID=strcat(context,lossi,'isUseWeights',string(loss.isUseWeights),'_isRemBKGND',string(loss.isRemoveBKGND),'_isAdapt',string(loss.isAdaptive));
    loss.lossi=lossi;
    [net,imdsTEST, pxdsTEST, timeTrain]=runSemanticSegmentationDICE(loss,expID,whichNet,...
        baseDataDir,inputSize,classNames,pixelLabelID,...
        nEpochs, lRate, isViewOnlyFirstChannel);



end  
    
