function retValue = ComputeMean(celldata,Prob,argName)
% Calculate the mean of cell matrix data; for example,
%  C = cell(m,n), where C{i,j} is a s x t matrix.
% Input: celldata (m=TESTNUMOUTER, n=TEST_num)
% Output: meanValue (numTestInner x Prob.N)
%   mean of celldata among `testOuter` datas

testSize = size(celldata);
numTestOuter = testSize(1);
numTestInner = testSize(2);

retValue = zeros(numTestInner,Prob.N);

for j = 1:Prob.N
    for testIn = 1:numTestInner
        average = zeros(numTestOuter,1);
        for testOut = 1:numTestOuter
            if ~exist('argName','var')
                average(testOut) = average(testOut) + mean(celldata{testOut,testIn}(j,:));
            elseif argName == "min"
                average(testOut) = average(testOut) + min(celldata{testOut,testIn}(j,:));
            elseif argName == "max"
                average(testOut) = average(testOut) + max(celldata{testOut,testIn}(j,:));
            end
        end
        retValue(testIn,j) = mean(average);
    end
end

end

