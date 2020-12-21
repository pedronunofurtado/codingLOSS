function data = convertCategoricalToNumeric(dataCategorical,...
                                            tablePattern,valueValue)
%a={{'A1', 'A11'}; {'A20','A11'}};
%ca=categorical(a{1});
%ca=[ca; categorical(a{2})];
%data = convertCategoricalToNumeric(ca,{'A1', 'A11','A20','A200'},[1 2 3 4]);

data=uint16(zeros(size(dataCategorical)));

for i=1:length(valueValue)
    data(dataCategorical==tablePattern{i})=valueValue(i);
end

end