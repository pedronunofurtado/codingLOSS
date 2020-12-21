function []=showEvaluationMetrics(whichShow,metrics,classNames)
% whichShow='all','confusion' 
metrics.ClassMetrics

metrics.ConfusionMatrix

metrics.NormalizedConfusionMatrix

normConfMatData = metrics.NormalizedConfusionMatrix.Variables;
figure
h = heatmap(classNames,classNames,100*normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title = 'Normalized Confusion Matrix (%)';

if(strcmp(whichShow,'all'))
    imageIoU = metrics.ImageMetrics.MeanIoU;
    figure
    h=histogram(imageIoU)
    title('Image Mean IoU')
end


end