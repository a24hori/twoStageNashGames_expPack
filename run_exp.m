%% run_exp.m
% =========================================================================
% Compare the results of the two methods:
%  - TSDRNEP: proposed model (with ambiguity of probability)
%  - TSSNEP: stochastic model (without ambiguity of probability)
% =========================================================================

%% Code
clear
clc
%rng(0,'twister'); % Mersenne-Twister (seed value)
% -------------------------------------------------------------------------
% Settings
eps = 0;
N = 2; % Number of players
K = 60; % Number of samples
xiDim = 3;
xiLower = -1;
xiUpper = 1;
TESTNUMOUTER = 30;
TEST_num = 21; % Number of instances
r = 0.8; % Regularization parameter for SVI
betagen.alpha = 1;
betagen.beta = 1;

A = 0.2.*diag(rand(N,1));
b = 1+rand(N,1);
c = 0.5+0.5.*rand(N,1);
alpha0 = 20;
beta0 = 2;
eta0 = 0.2+0.2.*rand(N,1);
zeta0 = 0.5+0.5.*rand(N,1);
s0 = 0.1+0.1.*rand(N,1);
% -------------------------------------------------------------------------

% Initialization
x.DRNE = cell(TESTNUMOUTER, TEST_num); % 1st stage variable (distirbutionally robust Nash eq., DRNE)
x.exPost = cell(TESTNUMOUTER, TEST_num); % 1st stage variable (Ex-post NE)
x.SAA = cell(TESTNUMOUTER, TEST_num); % 1st stage variable (SNE)
y.DRNE = cell(TESTNUMOUTER, TEST_num); % 2nd stage variable (DRNE)
y.exPost = cell(TESTNUMOUTER, TEST_num); % 2nd stage variable (ex post NE)
y.SAA = cell(TESTNUMOUTER, TEST_num); % 2nd stage variable (SNE)
lmd.DRNE = cell(TESTNUMOUTER, TEST_num); 
lmd.exPost = cell(TESTNUMOUTER, TEST_num);
lmd.SAA = cell(TESTNUMOUTER, TEST_num);
m_PWorst = cell(TESTNUMOUTER, TEST_num); % worst probability distribution in DRNE
P0Nominal = 1/K.*ones(K,1);

first_stage_prof.DRNE = cell(TESTNUMOUTER, TEST_num);%cell(TESTNUMOUTER, TEST_num);

recourse_function.inSample.DRNE = cell(TESTNUMOUTER, TEST_num);
recourse_function.outSample.DRNE = cell(TESTNUMOUTER, TEST_num);

second_stage_prof.inSample.DRNE = cell(TESTNUMOUTER, TEST_num);
second_stage_prof.outSample.DRNE = cell(TESTNUMOUTER, TEST_num);
second_stage_prof.worst.DRNE = cell(TESTNUMOUTER, TEST_num);

expected_profit.inSample.DRNE = cell(TESTNUMOUTER, TEST_num);
expected_profit.outSample.DRNE = cell(TESTNUMOUTER, TEST_num);
expected_profit.worst.DRNE = cell(TESTNUMOUTER, TEST_num);

DDAbsMax.inSample.DRNE = cell(TESTNUMOUTER, TEST_num);
DDAbsMax.outSample.DRNE = cell(TESTNUMOUTER, TEST_num);

now_time = datetime(now,'ConvertFrom','datenum');
fname = "./out/" + string(now_time.Year) + string(now_time.Month) + string(now_time.Day) +...
    "_" + string(now_time.Hour) + string(now_time.Minute) + "_" + "K=" + string(K) +...
    "_" + "N=" + string(N);
% -------------------------------------------------------------------------

fprintf("Run at %d/%02d/%02d %02d:%02d\n", now_time.Year, now_time.Month,...
    now_time.Day, now_time.Hour, now_time.Minute);


%% Conduct tests
for TESTOUTER = 1:TESTNUMOUTER

    %xi = xiLower + (xiUpper-xiLower).*rand(xiDim,K); % Generate K samples from N-dimensional normal distribution
    xi = xiLower + (xiUpper-xiLower).*betarnd(betagen.alpha,betagen.beta,xiDim,K);
    TEST = 1;
    
    parfor k=1:1
        % Start parallel computation
    end
    
    while TEST <= TEST_num
    
        fprintf("================= TEST num: %d =================\n", TEST);
        % Generate problem data
        rho0(1) = 0.1*(TEST-1);
        rho0(2) = 2-rho0(1);
        Prob = Problem(N,K,xi,r,rho0,A,b,c,alpha0,beta0,eta0,zeta0,s0);
        % Check if the variational inequality has a trivial solution.
    %     for l=1:K
    %         if all(Prob.veca<Prob.vecp(xi,l))
    %         else
    %             error("The numerical case has only a trivial solution.");
    %         end
    %     end
    
        % Compute TSDRNE by alternating PHA (A-PHA)
        tic
        [xDRNE, yDRNE, lmdDRNE, p_worst, state] = PHA_Alter(Prob);
        time_new = toc;
    
        if state == 1
            % Memo
            x.DRNE{TESTOUTER, TEST} = xDRNE;
            y.DRNE{TESTOUTER, TEST} = yDRNE;
            lmd.DRNE{TESTOUTER, TEST} = lmdDRNE;
            m_PWorst{TESTOUTER, TEST} = p_worst;
        
            % Display results of the test
            %fprintf("Comput. Time (Proposed): %d\n", time_new);
            %fprintf("=================================================\n");
            
            % ----------------------------------------------------------------
            % Compare expected profit function
            % ----------------------------------------------------------------
            
            recourse_function.inSample.DRNE{TESTOUTER, TEST} = zeros(N,K);
            recourse_function.outSample.DRNE{TESTOUTER, TEST} = zeros(N,K);
            
            % Generate 'out-sample' data (beta distribution)
            xiInSample = xi;
            xiOutSample = xiLower + (xiUpper-xiLower).*...
                betarnd(betagen.alpha+(-1+2*rand()),betagen.beta+(-1+2*rand()),xiDim,K);
        
            % Compute the expected profit of each player under the reference and worst probability
            for j=1:N
        
                % First stage profit
                first_stage_prof.DRNE{TESTOUTER, TEST}(j) = Prob.theta(j,xDRNE);
        
                % Compute recourse value for each scenario
                for k=1:K
                    recourse_function.inSample.DRNE{TESTOUTER, TEST}(j,k) =...
                        Prob.gamma(j,yDRNE,xiInSample,k);
                    recourse_function.outSample.DRNE{TESTOUTER, TEST}(j,k) =...
                        Prob.gamma(j,yDRNE,xiOutSample,k);
                end
                
                % Expected recourse under the worst-probability
                second_stage_prof.worst.DRNE{TESTOUTER,TEST}(j) =...
                    recourse_function.inSample.DRNE{TESTOUTER, TEST}(j,:)*p_worst(:,j);
                % First stage expected profit under the worst probability
                expected_profit.worst.DRNE{TESTOUTER,TEST}(j) =...
                    -first_stage_prof.DRNE{TESTOUTER,TEST}(j) - second_stage_prof.worst.DRNE{TESTOUTER,TEST}(j);
        
                % Compute sample average under out-sample verification data
                second_stage_prof.inSample.DRNE{TESTOUTER,TEST}(j) = ...
                    1/K*sum(recourse_function.inSample.DRNE{TESTOUTER,TEST}(j,:));
                second_stage_prof.outSample.DRNE{TESTOUTER,TEST}(j) =...
                    1/K*sum(recourse_function.outSample.DRNE{TESTOUTER,TEST}(j,:));
                % First stage expected profit with sample average
                expected_profit.inSample.DRNE{TESTOUTER,TEST}(j) =...
                    -first_stage_prof.DRNE{TESTOUTER,TEST}(j) - second_stage_prof.inSample.DRNE{TESTOUTER,TEST}(j);
                expected_profit.outSample.DRNE{TESTOUTER,TEST}(j) =...
                    -first_stage_prof.DRNE{TESTOUTER,TEST}(j) - second_stage_prof.outSample.DRNE{TESTOUTER,TEST}(j);
                
                % Compute directional derivative
                DDMax = DirectionalDerivative(recourse_function.inSample.DRNE{TESTOUTER,TEST}(j,:)',K,"max");
                DDMin = DirectionalDerivative(recourse_function.inSample.DRNE{TESTOUTER,TEST}(j,:)',K,"min");
                DDAbsMax.inSample.DRNE{TESTOUTER,TEST}(j) = max(abs(DDMax),abs(DDMin));
                DDMax = DirectionalDerivative(recourse_function.outSample.DRNE{TESTOUTER,TEST}(j,:)',K,"max");
                DDMin = DirectionalDerivative(recourse_function.outSample.DRNE{TESTOUTER,TEST}(j,:)',K,"min");
                DDAbsMax.outSample.DRNE{TESTOUTER,TEST}(j) = max(abs(DDMax),abs(DDMin));
            end
    
        else
            
            TEST = TEST - 1;
        
        end
    
        TEST = TEST + 1;
    
    end

end

%% ====================================================
% Compute average value in TESTNUMOUTER trials
% -----------------------------------------------------
aveProf.inSample.DRNE = zeros(TEST_num,N);
worstAveProf.inSample.DRNE = zeros(TEST_num,N);
aveProf.outSample.DRNE = zeros(TEST_num,N);
aveDD.inSample.DRNE = zeros(TEST_num,N);
aveDD.outSample.DRNE = zeros(TEST_num,N);
for j = 1:N
    for TEST = 1:TEST_num
        resAve = 0;
        resWorst = 0;
        resAveOut = 0;
        resDDin = 0;
        resDDout = 0;
        for TESTOUTER = 1:TESTNUMOUTER
            resAve = resAve + expected_profit.inSample.DRNE{TESTOUTER,TEST}(j);
            resWorst = resWorst + expected_profit.worst.DRNE{TESTOUTER,TEST}(j);
            resAveOut = resAveOut + expected_profit.outSample.DRNE{TESTOUTER,TEST}(j);
            resDDin = resDDin + DDAbsMax.inSample.DRNE{TESTOUTER,TEST}(j);
            resDDout = resDDout + DDAbsMax.outSample.DRNE{TESTOUTER,TEST}(j);
        end
        aveProf.inSample.DRNE(TEST,j) = resAve/TESTNUMOUTER;
        worstAveProf.inSample.DRNE(TEST,j) = resWorst/TESTNUMOUTER;
        aveProf.outSample.DRNE(TEST,j) = resAveOut/TESTNUMOUTER;
        aveDD.inSample.DRNE(TEST,j) = resDDin/TESTNUMOUTER;
        aveDD.outSample.DRNE(TEST,j) = resDDout/TESTNUMOUTER;
    end
end

% Compute the mean of lambda
xAve = ComputeMean(x.DRNE,Prob);
yAve = ComputeMean(y.DRNE,Prob);
yMinAve = ComputeMean(y.DRNE,Prob,"min");
yMaxAve = ComputeMean(y.DRNE,Prob,"max");
lmdAve = ComputeMean(lmd.DRNE,Prob);

% Compute the rate of change between xAve and yAve
RCValue = zeros(TEST_num,N);
for j = 1:N
    for test = 1:TEST_num
        RCValue(test,j) = (xAve(test,j)-yAve(test,j))/xAve(test,j);
    end
end

% Compute the marginal cost
dtheta = @(j,x) (Prob.matA(j,j)*x(j)+Prob.vecb(j));
MCValue = zeros(TEST_num,N);
for j = 1:N
    for test = 1:TEST_num
        MCValue(test,j) = dtheta(j,xAve(test,:)');
    end
end

%% ===================================================
% Profit function comparison
% ----------------------------------------------------
fprintf("===== Cost (-1*prifit) under in-sample average =====\n");
for TEST = 1:TEST_num
    fprintf("Test case %d:\n Player 1: %4f, Player 2: %4f\n", TEST, ...
        aveProf.inSample.DRNE(TEST,1), aveProf.inSample.DRNE(TEST,2));
end

fprintf("===== Cost (-1*profit) under worst-case mean =====\n");
for TEST = 1:TEST_num
    fprintf("Test case %d:\n Player 1: %4f, Player 2: %4f\n", TEST, ...
        worstAveProf.inSample.DRNE(TEST,1), worstAveProf.inSample.DRNE(TEST,2));
end

fprintf("===== Directional derivative (perturbation analysis) =====\n");
% Compute the maximum directional derivative Q^TdP at p subject to
%   sum(dP) = 0 
% where dP is the direction. This is because the perturbation requires that 
% p+dP in the polyhedron Δ:={p:sum(p)=1, p\geq 0}.
% Note that the problem above is formulated as
%   max         <Q,dP>
%   subject to  sum_{k=1}^K [dP]_k = 0,
%               \|dP\| = 1 (to ensure the boundedness of the problem).
for TEST = 1:TEST_num
    fprintf("--- Test case: %d --- \n", TEST);
    for j = 1:N
        fprintf("- Player %d. DDAbsMax: %4f\n", j, aveDD.inSample.DRNE(TEST,j));
    end
end

% fprintf("===== Kullback-Leibler divergence between P0 and P(:,j) =====\n");
% % Display KL-divergence between P0 and P
% for TEST = 1:TEST_num
%     fprintf("--- Test case: %d --- \n", TEST);
%     for j = 1:N
%         fprintf("- Player %d. KLdiv(P(:,%d)||P0): %4f\n", j, j, KLdiv(m_PWorst{TEST}(:,j),P0Nominal,K));
%     end
%     if N==2
%         fprintf("- KLdiv(P(:,1)||P(:,2)): %4f\n", KLdiv(m_PWorst{TEST}(:,1),m_PWorst{TEST}(:,2),K));
%         fprintf("- KLdiv(P(:,2)||P(:,1)): %4f\n", KLdiv(m_PWorst{TEST}(:,2),m_PWorst{TEST}(:,1),K));
%     end
% end

% for TEST = 1:TEST_num
%     figure
%     scatter(y.SAA{TEST}(1,:),y.SAA{TEST}(2,:),'o')
%     hold on
%     scatter(y.DRNE{TEST}(1,:),y.DRNE{TEST}(2,:),'x')
%     hold off
% end

%% ===================================================
% Plot profit curves
% ----------------------------------------------------
% Define marker
mark = ["b-x", "r-o"];
figure
hold on
for j = 1:N
    plot(0:0.1:0.1*(TEST_num-1),aveProf.inSample.DRNE(:,j),mark(j));
    %plot(0:0.1:0.1*(TEST_num-1),aveProf.outSample.DRNE(:,j));
end
hold off
title("Profit");
xlabel("KL measure \rho (\rho_{1}=\rho, \rho_{2}=2-\rho)");
ylabel("profit");
legend({'Player 1', 'Player 2'});
%legend({'P1 (in-sample)', 'P1 (out-sample)', 'P2 (in-sample)', 'P2 (out-sample)'});
% stackData = zeros(TEST_num,2,N);
% %stackData_outSample = stackData;
% %stackData_worst = stackData_outSample;
% legendData_SNE = cell(1,N);
% legendData_DRNE = cell(1,N);
% 
% for l=1:TEST_num
%    for j=1:N
%        stackData(l,1,j) = abs(expected_profit.inSample.SAA(j,l) - expected_profit.outSample.SAA(j,l));
%        stackData(l,2,j) = abs(expected_profit.inSample.DRNE(j,l) - expected_profit.outSample.DRNE(j,l));
%        % Make legend data
%        legendData_SNE{j} = "Player " + string(j) + " (SNE)";
%        legendData_DRNE{j} = "Player " + string(j) + " (DRNE)";
%    end
% end
% legendData = [legendData_SNE, legendData_DRNE];
% groupLabels = num2cell(1:TEST_num);
% 
% % Plot profit under the worst-case probability
% plotBarStackGroups(stackData,groupLabels);
% grid on
% legend(legendData);
% xlabel('Instance No.')
% ylabel('Difference')
% title("Diff. of Profit between in- and out-sample (K="+num2str(K)+")")
% exportgraphics(gcf,fname + ".eps")
% savefig(fname + ".fig")
% 
% for l=1:TEST_num
%    for j=1:N
%        stackData_outSample(l,1,j) = expected_profit.outSample.SAA(j,l);
%        stackData_outSample(l,2,j) = expected_profit.outSample.DRNE(j,l);
%        % Make legend data
%        legendData_SNE{j} = "Player " + string(j) + " (SNE)";
%        legendData_DRNE{j} = "Player " + string(j) + " (DRNE)";
%    end
% end
% legendData = [legendData_SNE, legendData_DRNE];
% groupLabels = num2cell(1:TEST_num);
% 
% 
% % Compute Kullback--Leibler divergence between in-sample distribution, or a
% % reference probability distribution, and out-sample distribution.
% % Q: uniform distribution => p_k=1/K k=1,...,K
% % P: beta distribution => 
% %KLval = KLdiv(P,Q,K);
% % Plot profit under reference probability
% plotBarStackGroups(stackData_outSample,groupLabels);
% grid on
% legend(legendData);
% xlabel('Instance No.')
% ylabel('Profit')
% title("Profit (K=" + num2str(K) + ")")
% exportgraphics(gcf,fname + "_outSample.eps")
% savefig(fname+"_outSample.fig");

% =======================================================
% Plot directional derivative curves
% -------------------------------------------------------
% In sample
figure
hold on
for j = 1:N
    plot(0:0.1:0.1*(TEST_num-1),aveDD.inSample.DRNE(:,j),mark(j));
end
hold off
title("Directional Derivative");
xlabel("KL measure \rho (\rho_{1}=\rho, \rho_{2}=2-\rho)");
ylabel("D.D. coeff.");
legend({'Player 1', 'Player 2'});

% Out sample
% figure
% for j = 1:N
%     plot(0:0.1:0.1*(TEST_num-1),aveDD.outSample.DRNE(:,j))
%     hold on
% end
% hold off
% title("Directional Derivative (out-sample)");
% xlabel("KL measure \rho_{1} (\rho_{2}=2-\rho_{1})");
% ylabel("D.D. coeff.");
% legend({'Player 1', 'Player 2'});

% =================================================================
% Marginal net revenue
% -----------------------------------------------------------------
figure
hold on
for j =1:N
    plot(0:0.1:0.1*(TEST_num-1),lmdAve(:,j),mark(j));
end
hold off
title("Marginal Revenue");
xlabel("KL measure \rho (\rho_{1}=\rho, \rho_{2}=2-\rho)");
ylabel("mean of \lambda_j(\xi_k)");
legend({'Player 1', 'Player 2'});
% ===================================================================

% ===================================================================
% Strategy curves
% -------------------------------------------------------------------
% first stage strategies
mark = ["b-x", "b-o", "r--x", "r--o"];
figure
hold on
for j = 1:N
    plot(0:0.1:0.1*(TEST_num-1),xAve(:,j),mark(j));
    plot(0:0.1:0.1*(TEST_num-1),yAve(:,j),mark(j+N));
end
hold off
title("First/second stage variables");
xlabel("KL measure \rho (\rho_{1}=\rho, \rho_{2}=2-\rho)");
ylabel("amount of production/supply");
legend({'Player 1 (prod.)', 'Player 1 (supp.)', 'Player 2 (prod.)', 'Player 2 (supp.)'});
% rate of change
mark = ["b-x", "r-o"];
figure
hold on
for j = 1:N
    plot(0:0.1:0.1*(TEST_num-1),RCValue(:,j),mark(j));
end
hold off
title("RC between x_j and the mean of y_j(\xi_k)");
xlabel("KL measure \rho (\rho_{1}=\rho, \rho_{2}=2-\rho)");
ylabel("Rate of Change (RC)");
legend({'Player 1', 'Player 2'});
% ===================================================================

% ===================================================================
% Marginal cost
% -------------------------------------------------------------------
figure
hold on
for j = 1:N
    plot(0:0.1:0.1*(TEST_num-1),MCValue(:,j),mark(j));
end
hold off
title("Marginal Cost");
xlabel("KL measure \rho (\rho_{1}=\rho, \rho_{2}=2-\rho)");
ylabel("\theta'_j(x_j)");
legend({'Player 1', 'Player 2'});

fin_time = datetime(now,'ConvertFrom','datenum');
fprintf("Finished at %d/%02d/%02d %02d:%02d\n", fin_time.Year, fin_time.Month, fin_time.Day,...
    fin_time.Hour, fin_time.Minute);

% Save numerical data
save(fname + ".mat");
% ========================================================