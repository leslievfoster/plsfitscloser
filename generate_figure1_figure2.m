% generate_figure1_figure2.m
%
% Generates Figures 1.1 and 1.2 in the paper 
%   "In the 2_norm, PLS Fits Closer Than or the Same As PCR"
%
% The code generates the image deblurring example.
%
% Leslie Foster, San Jose State University
% July 2026

% Created by Leslie Foster
% cc, Leslie Foster, 7-20-2026, leslie.foster@sjsu.edu


% This code was developed on an Apple M4 Pro with 48GB RAM.

rng(0)   % comment out this line for different random noise

%% create the example
fprintf('\nCreating the example\n')
NoiseLevel = 0.01;

n_pic = 100;   % The picture is n_pic by n_pic pixels
options.CommitCrime = 'on'; % Corollary 1.2 assumes A*x0 = b
%options.CommitCrime = 'off'; % little difference for k values in figures
[A, b, x0, ProbInfo] = PRblurspeckle(n_pic, options);
n = n_pic * n_pic;  % A is an n by n matrix stored in psf format
[bn, NoiseInformation]= PRnoise(b,'gauss',NoiseLevel);

% Convert A to a full matrix in order to use MATLAB's svd
ei = zeros(n,1);
Afull = zeros(n,n);
for i = 1:n
    ei(i) = 1;
    Afull(:,i) = A*ei;
    ei(i) = 0;
end
% Afull is an n by n full matrix

%check_Afull(A,Afull,b,bn,x0,NoiseLevel,options); %UNCOMMENT to check Afull

%% Calculate the pcr and pls solutions, with noise in rhs

% Calculate the singular value decomposition of Afull.
%    This required 1 minute on an Apple M4 pro
disp(' Starting the SVD. This can be slow. Potentially a minute.')
tic;
[U,s,V] = svd(Afull,"econ","vector"); % comment out, if svd already done
time_svd = toc;
fprintf('Time for  svd %8.2f seconds\n',time_svd)

% calculate the PCR (TSVD) solutions, with noise in rhs
k_pcr = n;
X_pcr = tsvd_b(U,s,V,bn,k_pcr);
% total error for pcr relative to norm(x0):
E_tot_pcr = sqrt(sum( (X_pcr - x0* ones(1,k_pcr) ).^2 )) /norm(x0);
% smallest total error
[ E_tot_pcr_min, k_tot_pcr_min ] = min(E_tot_pcr);

% Calculate the PLS solutions, with noise in rhs
k_pls = n/10;   % smaller dimension suffices for PLS

%k_pls = n;   % this option (UNCOMMENT line) can take several minutes

tic; fprintf('\nstarting bidiag2 for noisy rhs\n')
X_pls = bidiag2(A,bn,k_pls);     % call bidiag2
%time_pls_w_noise = toc

% total error for pls relative to norm(x0):
E_tot_pls = sqrt(sum( (X_pls - x0* ones(1,k_pls) ).^2 )) /norm(x0);
% smallest total error
[ E_tot_pls_min, k_tot_pls_min ] = min(E_tot_pls);


%% PLOT FOR FIGURE 1:
figure(1), clf

 subplot(2,2,1)
imagesc(reshape(x0, ProbInfo.xSize)), axis off, axis image, colormap(gray)
title('True solution','FontSize',30)
set(gca,'fontsize',30)
%
subplot(2,2,2)
imagesc(reshape(bn, ProbInfo.bSize)), axis off, axis image, colormap(gray)
title('Noisy data','FontSize',30)
set(gca,'fontsize',30)
%
subplot(2,2,3)
imagesc(reshape(X_pls(:,k_tot_pls_min), ProbInfo.xSize)), ...
    axis off, axis image, colormap(gray)
title(['PLS, k = ', num2str(k_tot_pls_min)],'FontSize',30)
set(gca,'fontsize',30)
%
subplot(2,2,4)
imagesc(reshape(X_pcr(:,k_tot_pls_min), ProbInfo.xSize)), ...
    axis off, axis image, colormap(gray)
title(['PCR, k = ',num2str(k_tot_pls_min)],'fontsize',30)
set(gca,'fontsize',30)

%% calculate the PCR and PLS solutions,  no noise in rhs

% calculate the PCR (TSVD) solutions, no noise in rhs
X_pcr_no_noise = tsvd_b(U,s,V,b,k_pcr);
% regularization error for pcr relative to norm(x0):
E_reg_pcr = sqrt(sum( (X_pcr_no_noise - x0* ones(1,k_pcr) ).^2 )) /norm(x0);

% Calculate the PLS solutions, no noise in rhs
tic, fprintf('\nstarting bidiag2 for noise free rhs\n')
X_pls_no_noise = bidiag2(A,b,k_pls);
%time_pls_no_noise = toc

% regularization error for pcr relative to norm(x0):
E_reg_pls = sqrt(sum( (X_pls_no_noise - x0* ones(1,k_pls) ).^2 )) /norm(x0);


%% PLOT FOR FIGURE 2:

% find last index where E_tot_pcr < 1.
k_E_pcr_lt_1 = find(E_tot_pcr < 1); k_E_pcr_lt_1 = k_E_pcr_lt_1(end);
% find last index where E_tot_pls < 1.
k_E_pls_lt_1 = find(E_tot_pls < 1); k_E_pls_lt_1 = k_E_pls_lt_1(end);

%k_E_pcr_lt_1 = k_pcr; k_E_pls_lt_1 = k_pls; % UNCOMMENT to plot more pts

figure(2), clf

semilogx(...
1:k_E_pcr_lt_1, E_tot_pcr(1:k_E_pcr_lt_1), '--<r', ...
1:k_E_pls_lt_1, E_tot_pls(1:k_E_pls_lt_1), '-<b',...
1:k_E_pcr_lt_1, E_reg_pcr(1:k_E_pcr_lt_1),'-->m',...
1:k_E_pls_lt_1, E_reg_pls(1:k_E_pls_lt_1),'->c',...
'MarkerSize',9)

axis([0,n,0,1])  
xtickformat('%10d')
xticklabels(xticks)    % this command creates xlabels without exponents
grid
title(['Semiconvergence History of PCR and PLS for the PRblurspeckle Example'])
graph_legend =...
{ ['Total Error PCR / ||x_0||, min = ',num2str(E_tot_pcr_min,3),...
   ' at k = ',num2str(k_tot_pcr_min)], ...
['Total Error PLS  / ||x_0||, min = ',num2str(E_tot_pls_min,3),...
 ' at k = ',num2str(k_tot_pls_min)], ...
['Regularization Error PCR / ||x_0|| = ',...
 num2str(E_tot_pcr(k_tot_pls_min),3),' at k = ',num2str(k_tot_pls_min)],...
['Regularization Error PLS  / ||x_0|| = ',...
 num2str(E_tot_pls(k_tot_pls_min),3),' at k = ',num2str(k_tot_pls_min)],...
};
legend(graph_legend,'location','best')
ylabel('2-Norm of Total Errors and Regularization Errors / ||x_0||')
xlabel(' Subspace Dimension k')
set(gca,'FontSize',22)

% the following text locations are designed for example in paper
text(35,.295,['Best  ';'PLS,  ';['k = ',num2str(k_tot_pls_min,2)];...
    '|     ';'v     '],'FontSize',22)
text(2800,.31,['Best   ';'PCR,   ';['k= ',num2str(k_tot_pcr_min,4)];...
    '      |';'      v'],'FontSize',22)
text(100,0.075,['         \^   ';'PLS Reg. Error'],'FontSize',22)
text(1200,0.140,['PCR Reg. Error  >'],'FontSize',22)
shg

%% ========================================================================
% Function bidiag2 to calculate pls solutions
%==========================================================================
function [X,W,T,B] = bidiag2(A,b,k)
% Calculates the PLS approximate solutions to min ||A x - b|| for Krylov
%     subspace dimensions i = 1 to k. The ith Krylov subspace is 
%     K_i = span(A'b, (A'A) A' b, (A'A)^2 A' b, ... (A'A)^(i-1) A' b)
% Usage: X = bidiag2(A,b,k) or [X,W,T,B] = bidiag2(A,b,k)
% Input:
%    A  -  m by n matrix
%    b  -  m by 1 vector
%    k  - integer, 0 <= k <= min(m,n)
% Output:
%    X - an n by k matrix where X(:,i) is the PLS approximate
%        solution for subspace dimension i
%    W - the "weights", W is an orthonormal basis for K_k
%    T - the "scores", T is an orthonormal basis for A * K_k
%    B - bidiag matrix, B stored by diagonals 
%        B (when converted) = T' * A * W, 
%    After completion, consider conversions
%    B_full = diag(B(:,1)) + diag( B(1:k-1,2),1 ) ; %or
%    B_sp = spdiags(B, [0, 1], k, k+1); B_sp = B_sp(1:k,1:k);

% --------------------------------------------------------------------
% --------------------- Ake Bjorck 9/6-2016 --------------------------
% --                 Source: References [2] and [3]                 --
% --------------------------------------------------------------------
% This code is from [3] with the exit condition in [2] added.
%
% This code does not center A and b initially. To do this add code:
%     A = A - ones(m,1) * mean(A); b = b - mean(b);

B = zeros(k,2);   % B stored by diagonals
w = A'*b;  w = w/norm(w); W = w;
t = A*w;  rho = norm(t);  t = t/rho; T = t;
B(1,1) = rho;
d = w/rho; X = (t'*b)*d;
% ---------------Continue bidiagonalization ---------
for i = 2:k
    w = A'*t - rho*w; w = w - W*(W'*w);  % Reorthogonalize w
    theta = norm(w);  w = w/theta;  W(:,i) = w;
    t = A*w - theta*t; t = t - T*(T'*t);   % Reorogonalize t
    rho = norm(t); B(i-1,2) = theta; B(i,1) = rho;
    if rho == 0
        error(['The algorithm failed for i =',num2str(i)]);
        % unlikely in finite precision arithmetic
    end
    t = t/rho; T(:,i) = t;
% ----------- Update regression coefficeints ---------------
    d = (w - theta*d)/rho;
    X(:,i) = X(:,i-1) + (t'*b)*d;
end
end  %end of function

%% ========================================================================
% Function tsvd_b to calculate pcr / tsvd solutions
%==========================================================================
function X =tsvd_b(U,s,V,b,k)
% Calculates pcr / tsvd solutions to min || A x - b ||, with A m x n 
% Use: 
%    [U,s,V]=svd(A,'econ','vector'); X = tsvd_b(U,s,V,b,k);
% Input:
%    U - left singular vectors of A, m by min(m,n) matrix
%    s - singular values of A, column vector, min(m,n) by 1 
%    V - right singular vectors of A, n by min(m,n) matrix
%    b - right hand side vector, m by 1
%    k - number of tsvd solutions, 1 <= k <= min(m,n)
% Output: n by k matrix X 
%    X(:,j), 1 <= j <= k, is the truncated singular value / principal
%    component regression approximate solution to min || A x - b || when
%    using the j largest singular values and their singular vectors
% 
n = size(V,1);
m = size(U,1);
if  ( k < 0 ) | ( min(m,n) < k ) 
  error('Illegal truncation parameter k')
end
if size(s,2) > 1 % if s is a matrix, assume singular values are in diag(s)
    s=diag(s);
end
z  = U(:,1:k)' * b;
y = z ./ s(1:k);
X = cumsum( V(:,1:k) .* (y(1:k)') , 2);
% This code is a faster version of 
% X = cumsum( V(:,1:k) .* (ones(n,1)*y(1:k)') , 2);  % or of
% for j = 1:k, X(:,j) = V(:,1:j) * ( ( U(:,1:j)' * b) ./ s(1:j)) ; end
end     % end of function

%% ========================================================================
% Function that has some checks that Afull is correct
%==========================================================================
function check_Afull(A,Afull,b,bn,x0,NoiseLevel,options)
% tests that the linear operator for the psf matrix A is same as
% the linear operator for the double matrix Afull
[m,n] = size(Afull);
fprintf('\nSome checks that Afull and A represent same linear operator:\n')
if strcmp(options.CommitCrime,'on')
    fprintf(...
       ['    norm(Afull*x0 - b)/norm(b)  = %10.3e. ',...
       'It should be numerically zero. \n'],...
        norm(Afull*x0 - b)/norm(b))
    fprintf(...
       ['    norm(Afull*x0 - bn)/norm(b) = %10.3e. ', ...
       'It should be %10.3e . \n'],...
        norm(Afull*x0 - bn)/norm(b),NoiseLevel )
end
fprintf('    norm(A * x - Afull*x)/ ( norm(x) * ||A||_1 ) for random x, \n')
fprintf('    should be numerically zero: \n')
normA1 = norm(Afull,1);
nrep = 5;
for i = 1:nrep
    x = randn(n,1);
    rel_error = norm(A * x - Afull*x)/ (norm(x) * normA1);
    fprintf('     %8.3e\n',rel_error)
end
fprintf('\n')
end     % end of function





