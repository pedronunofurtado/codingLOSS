
%-------------------------------- MRI ------------------------

pixelLabelID=[0, 63, 126,189,252]; %];
baseDataDir=DUAL_TWICEdir;
inputSize=[256 256 3];
classNames=["BackGround",...
           "liver",...
           "spleen",...
           "rkidney",...
           "lkidney"];
context='MRIthree';        
whichNet='DEEPLAB';nEpochs=200;lRate=0.005;

%-------------------------------- CT ------------------------


nepochs=300;
learnRate=0.0005;
expID="1_CT_DEEPLAB"
whichNet='DEEPLAB';
pixelLabelID=[]; % for autoassignment 0..nClasses
baseDataDir = CTdir;
inputSize=[512 512 3];
classNames=["BackGround",...
            "liver"];


%------------------------------ IDRID ----------------------
pixelLabelID=[];
baseDataDir=IDRID_TWICEdir;
%baseDataDir=IDRIDreinforceDIR;
inputSize=[512 1024 3];
classNames=[
    "Background"
    "Microaneurysms"
    "Haemorrhages"
    "HardExudates"
    "SoftExudates"
    "OpticDisc"
    ];
context='IDRIDthree';  
whichNet='DEEPLAB';lRate=0.005;
nEpochs=300;

end