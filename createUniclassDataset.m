function createUniclassDataset(baseDataDir,classID, id)
%createUniclassDataset:createUniclassDataset('F:\Pedro\IDRID\a\IDRIDdataset\Original\A. Segmentation\RESIZED5121024','MA',1);
%createUniclassDataset('F:\Pedro\downloads\ChaosOrganSeg\CHAOS_Train_Sets\Train_Sets\MR\SEMT1DUAL','L',63);
%createUniclassDataset('F:\Pedro\downloads\ChaosOrganSeg\CHAOS_Train_Sets\Train_Sets\MR\SEMT1DUAL','S',126);
%'RK',189
%'LK',252

%gets all gndtruth images,converts each to 0,1 where 0 is from 0,
%%1 is from id; stores as new dir and return the dir

mclass='\3. MultiClass Groundtruths'

dIN= {strcat(baseDataDir,mclass,'\a. Training Set'),
       strcat(baseDataDir,mclass,'\b. Testing Set'), 
       strcat(baseDataDir,mclass,'\b. Testing SetTEST'), 
       strcat(baseDataDir, mclass,'\b. Testing SetVAL') }; 

dOUT= {strcat(baseDataDir,'\',classID,'\a. Training Set'),
       strcat(baseDataDir,'\',classID,'\b. Testing Set'), 
       strcat(baseDataDir,'\',classID,'\b. Testing SetTEST'), 
       strcat(baseDataDir,'\', classID,'\b. Testing SetVAL') };

mkdir(dOUT{1});
mkdir(dOUT{2});
mkdir(dOUT{3});
mkdir(dOUT{4});

for i=1:length(dIN)
    files=dir(dIN{i})
    for k=1:length(files)
        fileName=files(k).name
        if(startsWith(fileName,'.'))
            continue
        end
        im=imread( strcat(dIN{i},'\',fileName) );

        im(im~=id)=0;
        im(im==id)=1;
        imwrite(im,strcat(dOUT{i},'\',fileName));
    end 

end



