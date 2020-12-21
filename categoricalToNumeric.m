function A=categoricalToNumeric(C)

A=zeros(size(C));

uv=unique(C);

for i=1:length(uv)
    A(C==uv(i))=i-1;
end

end