function []=evalSemanticSegmentationNetSave(net,classNames,...
                                    imdsTEST, pxdsTEST,...
                                    writeName,...
                                    isSaveResults,resultsDir)

%Run the network on the test images. Predicted labels are written to disk in a temporary directory and returned as a pixelLabelDatastore object.
writeDir=strcat(resultsDir,'/semanticsegResults','_',writeName);
mkdir(writeDir);

pxdsResults = semanticseg(imdsTEST,net,'MiniBatchSize',4,"WriteLocation",writeDir);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTEST);

showEvaluationMetrics('confusion',metrics,classNames);

%metrics.ClassMetrics
%metrics.ConfusionMatrix

normConfMatData = metrics.NormalizedConfusionMatrix.Variables;

if(isSaveResults)
    whichName=strrep(writeDir,"/","_");

    save(strcat(resultsDir,'/metrics_',whichName),'metrics');    
    save(strcat(resultsDir,'/normConf_',whichName),'normConfMatData');
end


end