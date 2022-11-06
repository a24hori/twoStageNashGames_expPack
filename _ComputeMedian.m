function meanValue = ComputeMedian(celldata,numTestInner,numTestOuter,Prob)
% Calculate the mean of cell matrix data; for example,
%  C = cell(m,n), where C{i,j} is a s x t matrix.
% Input: celldata (m=TESTNUMOUTER, n=TEST_num)
% Output: meanValue (numTestInner x Prob.N)
%   mean of celldata among `testOuter` datas

meanValue = zeros(numTestInner,Prob.N);

for j = 1:Prob.N
    for testIn = 1:numTestInner
        average = zeros(numTestOuter,1);
        for testOut = 1:numTestOuter
            average(testOut) = average(testOut) + mean(celldata{testOut,testIn}(j,:));
        end
        meanValue(testIn,j) = mean(average);
    end
end

end

