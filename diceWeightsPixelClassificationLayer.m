classdef diceWeightsPixelClassificationLayer < nnet.layer.ClassificationLayer
    % This layer implements the generalized Dice loss function for training
    % semantic segmentation networks.
    
    properties %(Constant)
        % Small constant to prevent division by zero. 
        Epsilon = 1e-8;  
        W;
        classWeights;
        miniBatchSize;
    end
    
    methods
        
        function layer = diceWeightsPixelClassificationLayer(name,classWeights, miniBatchSize)
            % layer =  dicePixelClassificationLayer(name) creates a Dice
            % pixel classification layer with the specified name.
            
            % Set layer name.          
            layer.Name = name;

            layer.classWeights=classWeights;
            layer.miniBatchSize=miniBatchSize;
            
            W1=zeros(1,1,length(classWeights),miniBatchSize);
            for i=1:miniBatchSize
                W1(1,1,:,i)=classWeights;
            end
            
            layer.W=W1;
            
            % Set layer description.
            layer.Description = 'Dice loss';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Dice loss between
            % the predictions Y and the training targets T.   

            % Weights by inverse of region size.
            W = layer.W;
                    
            if(size(T,4)~=layer.miniBatchSize)
                W1=zeros(1,1,length(layer.classWeights),size(T,4));
                if(~isempty(layer.classWeights)) % empty means adaptive
                    for i=1:size(T,4)
                        W1(1,1,:,i)=layer.classWeights;
                    end
                end
                W=W1;
            end            
            
            intersection = sum(sum(Y.*T,1),2);
            union = sum(sum(Y.^2 + T.^2, 1),2);          
            
            numer = 2*sum(W.*intersection,3) + layer.Epsilon;
            denom = sum(W.*union,3) + layer.Epsilon;
            
            % Compute Dice score.
            dice = numer./denom;
            
            % Return average Dice loss.
            N = size(Y,4);
            loss = sum((1-dice))/N;
            
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the Dice loss with respect to the predictions Y.
            
            % Weights by inverse of region size.
            W = layer.W;

             if(size(T,4)~=layer.miniBatchSize)
                W1=zeros(1,1,length(layer.classWeights),size(T,4));
                if(~isempty(layer.classWeights)) % empty means adaptive
                    for i=1:size(T,4)
                        W1(1,1,:,i)=layer.classWeights;
                    end
                end
                W=W1;
            end             
            
            intersection = sum(sum(Y.*T,1),2);
            union = sum(sum(Y.^2 + T.^2, 1),2);
     
            numer = 2*sum(W.*intersection,3) + layer.Epsilon;
            denom = sum(W.*union,3) + layer.Epsilon;
            
            N = size(Y,4);
      
            dLdY = (2*W.*Y.*numer./denom.^2 - 2*W.*T./denom)./N;
        end
    end
end