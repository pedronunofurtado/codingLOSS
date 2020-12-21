function [TPR,FPR]=rocRatesMaxPCT(lmapPREV, lmap, classIDs, maxPCT)

P=countEachInMatrix(lmap);

if(size(P,2)~=length(classIDs))
    for i=1:length(classIDs)
        if(P(i,1)~=classIDs(i))
            P=[P(1:i-1,:); classIDs(i) 0; P(i:end,:)];
        end    
    end
end

Pc=P(:,1);
Pv=P(:,2);

Nv=Pv*(maxPCT);


TPR=zeros(size(Pc,1),1);
FPR=zeros(size(Pc,1),1);

for i=1:length(Pc)
   TPi = sum( sum(lmapPREV==Pc(i) &  lmap==Pc(i) ) );
   FPi = sum(sum(lmapPREV==Pc(i) &  lmap~=Pc(i) ) );
   
   if(Pv(i)>0)
    TPR(i)=double(TPi)/double(Pv(i));
   else
    TPR(i)=0;
   end
   
   if(Nv(i)>0)
    FPR(i)=double(FPi)/double( Nv(i) );
   else
    FPR(i)=0;
   end
   
   if(TPR(i)>1) TPR(i)=1; end
   if(FPR(i)>1) FPR(i)=1; end   
end



end