function plotRoc(fpr,tpr, className, isScatter)

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
    
    %plot
    if(isScatter)
        figure, scatter(fpr,tpr);
    else
        [fpr,orderSort]=sort(fpr);
        tpr=sort(tpr);
        %tpr=tpr(orderSort);
        figure, plot(fpr,tpr);
    end
    
    title(className);%title(num2str(i));
    xlabel('False Positive Rate');
    ylabel('True Positive Rate')
    ylim([0 1]);
    %xlim([0 1]); 

end