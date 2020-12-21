function subplotRoc(fpr,tpr, classNames, isScatter, plotsMtx)

figure;


for i=1:length(classNames) %size(fpr,2)
    subplotRoci(i,fpr(:,i),tpr(:,i), classNames(i), isScatter, plotsMtx);
end

end

function subplotRoci(i,fpr,tpr, className, isScatter, plotsMtx)

%plotsMtx=[2 3]
    % remove repeated 1 by using smallest tpr in 1
%     idx=find(fpr==1);
%     if(~isempty(idx))
%         keep=tpr(idx);
%         keepi=find(keep==max(keep));
%         a=keep(keepi);
%         a=a(1);
%         tpr(idx)=[]; fpr(idx)=[];
%         tpr(end+1)=a; fpr(end+1)=1;
%     end

xy2=[0,1];

    %plot
    if(isScatter)
        subplot(plotsMtx(1),plotsMtx(2),i), scatter(fpr,tpr);
    else
        [fpr,orderSort]=sort(fpr);
        tpr=sort(tpr);
        %tpr=tpr(orderSort);
        if(fpr(end)==1)
            subplot(plotsMtx(1),plotsMtx(2),i), plot(fpr,tpr,xy2,xy2,'--','LineWidth',3);
        else
            subplot(plotsMtx(1),plotsMtx(2),i), plot(fpr,tpr,'LineWidth',3);
        end
    end
    
    title(className);%title(num2str(i));
    xlabel('False Positive Rate');
    ylabel('True Positive Rate')
    ylim([0 1]);
    %xlim([0 1]); 

end