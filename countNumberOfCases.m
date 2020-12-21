function nLabelsTotal=countNumberOfCases(filesDir)
imds = imageDatastore(fullfile(filesDir), ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');    

nLabelsTotal=0;
for i=1:length(imds.Files)
    img = readimage(imds,i);

    bwimg=bwlabel(img);
    nLabels=length(unique(bwimg))-1;
    
    nLabelsTotal=nLabelsTotal+nLabels;
end



disp(nLabelsTotal);
end
    