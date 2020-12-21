classdef tverskyPixelClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the Tversky loss function for training
    % semantic segmentation networks.
    
    % References
    % Salehi, Seyed Sadegh Mohseni, Deniz Erdogmus, and Ali Gholipour.
    % "Tversky loss function for image segmentation using 3D fully
    % convolutional deep networks." International Workshop on Machine
    % Learning in Medical Imaging. Springer, Cham, 2017.
    % ----------
    
    
    properties(Constant)
        % Small constant to prevent division by zero.
        Epsilon = 1e-8;
    end
    
    properties
        % Default weighting coefficients for False Positives and False
        % Negatives
        Alpha = 0.5;
        Beta = 0.5;
        classWeights; % 1xnumClasses
        W; % 1x1xnumClassesxminiBaychSize
        numClasses;
        miniBatchSize;
    end

    
    methods
        
        function layer = tverskyPixelClassificationLayer(name, alpha, beta, classWeights, miniBatchSize)
            % layer =  tverskyPixelClassificationLayer(name, alpha, beta) creates a Tversky
            % pixel classification layer with the specified name and properties alpha and beta.
            
            % Set layer name.          
            layer.Name = name;
            
            layer.Alpha = alpha;
            layer.Beta = beta;
            
            layer.numClasses=length(classWeights);
            layer.miniBatchSize=miniBatchSize;
            
            layer.classWeights=classWeights;
            
            W1=zeros(1,1,length(layer.classWeights),miniBatchSize);
            if(~isempty(classWeights)) % empty means adaptive
                for i=1:miniBatchSize
                    W1(1,1,:,i)=classWeights;
                end
            end
            layer.W = W1;
            
            % Set layer description.
            layer.Description = 'Tversky loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Tversky loss between
            % the predictions Y and the training targets T.   

            Pcnot = 1-Y;
            Gcnot = 1-T;
            TP = sum(sum(Y.*T,1),2);
            FP = sum(sum(Y.*Gcnot,1),2);
            FN = sum(sum(Pcnot.*T,1),2); 
            
            numer = TP + layer.Epsilon;
            denom = TP + layer.Alpha*FP + layer.Beta*FN + layer.Epsilon;
          
            IoU = numer./denom;
            
            classWeightsi=layer.W;
            
            if(size(T,4)~=layer.miniBatchSize)
                W1=zeros(1,1,length(layer.classWeights),size(T,4));
                if(~isempty(classWeightsi)) % empty means adaptive
                    for i=1:size(T,4)
                        W1(1,1,:,i)=layer.classWeights;
                    end
                end
                classWeightsi=W1;
            end
            
            %adaptive classWeights: worst IoU is multiplied by n (classes)
            % second worst by n-1 and so on...
            if(isempty(layer.classWeights)) % use adaptive class weights
                classWeightsi=zeros(size(IoU));
                for i=1:size(T,4)
                    [B,I]=sort(extractdata(IoU(1,1,:,i)));
                    for j=1:length(I)
                        classWeightsi(1,1,I(j),i) = length(I)-j; 
                    end
                end
            end
            
            
            % Compute tversky index
            %lossTIc = 1 - numer./denom;
            lossTIc = 1 - IoU;
            %lossTI = sum(lossTIc,3);
            lossTI = sum(classWeightsi.*lossTIc,3);
            
            % Return average tversky index loss.
            %N = size(Y,4);
            N = sum(sum(classWeightsi));
            loss = sum(lossTI)/N;

        end
    end
end