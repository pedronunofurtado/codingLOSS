function transformFilesIntoBinary(filesDir)
%transformFilesIntoBinary('/Users/pedro/Documents/MATLAB/Data/2020/diaretdb1/baseDraft/');
imds = imageDatastore(fullfile(filesDir), ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');    


for i=1:length(imds.Files)
    img = readimage(imds,i);
    
    img=keepImportantRegionsV3(img, 20, 500);
        
    img=uint8(img);
    imwrite(img,imds.Files{i});
end

end  