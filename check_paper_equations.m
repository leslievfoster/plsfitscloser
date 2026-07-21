function check_equations(m, n, k) 
%==========================================================================
% MATLAB program to check the equations and formulas in
% the paper "IN THE 2-NORM, PLS FITS CLOSER THAN OR SAME AS PCR"
% for specific values of m, n, and k:
% Input Parameters: m, n, k where
%        for min || A x - b ||
%    A is m by n diagonal matrix with symbolic entries
%    b is an m by 1 symbolic vector
%    k is the PLS and PCR subspace dimension.
%      1 <= k <= min(m,n) is required
% To use with the default values m = 5, n = 4, k = 2:
%    check_paper_equations
% To use for an underdetermined A (e.g., m = 4, n = 5, k = 2):
%    check_paper_equations(4,5,2)
% For specific values of m, n, and k the program uses MATLAB's
%    computer algebra system to check that the equations in
%    the paper, as coded in the program, appear to be correct.
%    The purpose is to detect, for specific values of m, n, and k,
%    if there are typos or other errors in the paper's equations.
%==========================================================================

% The code checks the equations in the paper for diagonal A. This
%     is also a check for general A, since Theorem 1.1 and Corollary 1.2
%     are true for general A in (1.1) with singular value decomposition
%     A = U D V^T, if and only if the results are true for min||Dx - U'*b||
% Due to the resource demands for Matlab's symbol manipulation, the program
%     is practical for smaller values (e.g., k <= 3) of m, n, k. For
%     larger values the program can be slow, stall, or fail.
% This program should not be considered a formal verification of the
%     correctness of the paper's formulas and equations.
%
% Created by Leslie Foster with the assistance of Google's Gemini
% cc, Leslie Foster, 7-9-2026, leslie.foster@sjsu.edu

% Default arguments
    arguments
        m (1,1) {mustBeInteger} = 5
        n (1,1) {mustBeInteger} = 4
        k (1,1) {mustBeInteger} = 2
    end

fprintf('====================================================\n');
fprintf('       SYMBOLIC EQUATION CHECKS FOR           \n');
fprintf('''IN THE 2-NORM, PLS FITS CLOSER THAN OR SAME AS PCR''\n')
fprintf('====================================================\n\n');


fprintf('Checks are for m = %d, n = %d, and k = %d.\n\n', m, n, k);

%% 1. Checks dimensions

if ~( (1 <= k) & ( k<= min(m,n) ) )
    disp('1 <= k <= min(m,n) is required')
    return
end

if k == 1
    disp('*****>  For k = 1 and min(m,n) > 20, program may be slow, stall, or fail.')
    disp('        Ctrl c to exit.')
    disp(' ')
elseif (k == 2) & (min(m,n) > 6)
    disp('*****>  For k = 2 and min(m,n) > 6, program may be slow, stall, or fail.')
    disp('        Ctrl c to exit.')
    disp(' ')
elseif (k == 3) & (min(m,n) > 4)
    disp('*****>  For k = 3 and min(m,n) > 4, program may be slow, stall, or fail.')
    disp('        Ctrl c to exit.')
    disp(' ')
elseif k > 3
    disp('*****>  For k >= 4, program may be slow, stall, or fail.')
    disp('        Ctrl c to exit.')
    disp(' ')
end
% The practical limits on m, n, and k depend on the computer being used.

%% 2. Define independent symbolic variables
s = sym('s', [1, n], 'real'); % Singular values (s_1, ..., s_n)
b = sym('b', [m, 1], 'real'); % Right-hand side vector (b_1, ..., b_m)
if m < n    % underdetermined case
   re = sym('re', [n-m, 1], 'real'); % for arbitrary undetermined
         % components in a solution to A c = b
   s(m+1:n)=0;    % convention for singular values for m < n
end
A = sym(zeros(m, n));
for i = 1:min([m n])
    A(i,i) = s(i);
end

%% 3. Compute exact ordinary least squares (OLS) solution c
% Satisfies normal equations (1.2): A'*A*c = A'*b
t = s.^2; % t_i = s_i^2
if m >= n
   c = (A'*A) \ (A'*b);  
elseif m < n
    % for m < n, (A'*A) c = A'*b is undetermined
    % MATLAB issues a warning, which we suppressed
    warnState = warning('off', 'symbolic:mldivide:RankDeficientSystem');
    c = (A'*A) \ (A'*b);  
    warning(warnState); % Restore the original warning state    
    % MATLAB chooses a specific solution, settings the last n-m 
    % components of c to 0
    % We create a general solution by setting the
    % last n-m components of c to a new symbolic vector,
    c(m+1:n) = sym('und', [n-m, 1], 'real'); 
end
% Since A is diagonal, V = eye(n), so d = V'*c = c
d = c;
w = d.^2; % w_i = d_i^2

%% 4. Generate Moments m_j for j = 0 to 2*k+2
max_moment = 2*k + 2; 
moments = sym(zeros(max_moment + 1, 1));
for j = 0:max_moment
    % Equation (2.2): m_j = sum_{i=1}^n w_i * t_i^j
    moments(j+1) = sum(w .* (t.^j)','all');
end

% Helper function to fetch moment m_j (using 0-based indexing)
m_val = @(j) moments(j+1);

%% 5. Construct Hankel Matrix H with exact dimensions (k+2) x (k+2)
H_dim = k + 2; 
H = sym(zeros(H_dim, H_dim));
for r = 1:H_dim
    for col = 1:H_dim
        H(r,col) = m_val(r + col - 2); % Max moment accessed is (k+2)+(k+2)-2 = 2*k+2
    end
end

% Extract submatrices using paper definitions
B = H(2:k+1, 3:k+2);                      % Eq (2.4) / text
C = H(2:k+1, 2:k+1);                      % Eq (2.4) / text
a = H(3:k+2, 1);                          % Eq (2.5) -> [m_2, ..., m_{k+1}]'
e = H(2:k+1, 1);                          % Eq (2.5) -> [m_1, ..., m_{k}]'
D = H(1:k+1, 2:k+2);                      % Eq (3.2)
E = H(2:k+1, [2, 4:k+2]);                 % Eq (3.2)
M = [sum(w(1:k)), e'; a, B];              % Eq (4.1)

%% 6. Generate Krylov Subspace Solutions (PLS)
Kk = sym(zeros(n, k));
current_vec = A'*b;
for j = 1:k
    Kk(:, j) = current_vec;
    current_vec = (A'*A) * current_vec;
end
% Explicitly use normal equations to solve the symbolic overdetermined system
alpha_krylov = (Kk'*A'*A*Kk) \ (Kk'*A'*b); 
z_k = Kk * alpha_krylov;

%% 7. Generate Principal Component Regression Solutions (PCR)
% For diagonal A, columns of V are standard basis vectors
y_k = c; 
y_k(k+1:end) = 0; % Keep only first k components

%also calculate as solution to least squares with restricted domain (1.4) 
Pk = sym(eye(n,k));
alpha_pcr = (Pk'*A'*A*Pk) \ (Pk'*A'*b);
y_k_Pk = Pk*alpha_pcr;


%% 8. Solve for alpha using Moment Equation (2.6)
alpha = B \ a;


%% ------------------------------------------------------------------------
% SECTION 2 EQUATION CHECKS
%--------------------------------------------------------------------------

% Eq 2.6: Moment equation for alpha vs alpha via Krylov
check_eq(2.6, alpha - alpha_krylov, 'Moment equation for alpha vs alpha via Krylov(2.6)');

% Eq 2.7: PLS Error Formulation (Left Identity)
check_eq(2.7, (z_k - c).'*(z_k - c) - (m_val(0) - 2*alpha'*e + alpha'*C*alpha), 'PLS error formula LHS (2.7)');

% Eq 2.7: PLS Error Formulation (Right Breakdown Identity)
alpha_e_alphap_C_alpha = alpha'*e - alpha'*C*alpha ; % save for repeated use
check_eq(2.7, (m_val(0) - 2*alpha'*e + alpha'*C*alpha) - (sum(w) - alpha'*e - alpha_e_alphap_C_alpha), 'PLS error formula RHS breakdown (2.7)');

% Eq 2.8: Theoretical y_k vs y_k by least squares
check_eq(2.8, y_k - y_k_Pk, 'Theoretical y_k vs y_k via least squares (1.4)');

% Eq 2.8: PCR Error formulation
%check_eq(2.8, norm(y_k - c)^2 - sum(w(k+1:n)), 'PCR error formula (2.8)');
check_eq(2.8, (y_k - c).'*(y_k - c) - sum(w(k+1:n)), 'PCR error formula (2.8)');

% Eq 2.9: Main Theorem Difference Formulation
lhs_6 = (y_k - c).'*(y_k - c) - (z_k - c).'*(z_k - c);
rhs_6 = alpha_e_alphap_C_alpha + (-sum(w(1:k)) + alpha'*e);
check_eq(2.9, lhs_6 - rhs_6, 'Difference in PCR and PLS errors (2.9)');

%% ------------------------------------------------------------------------
% SECTION 3 EQUATION CHECKS (Stieltjes & Determinants)
%--------------------------------------------------------------------------
detE=det(E); % to only calculate once
detD=det(D);
detB=det(B);
% Eq 3.4: Vector Subspace Residual Identity
check_eq(3.4, (e - C*alpha) - [m_val(1) - a'*alpha; zeros(k-1,1)], 'Subspace residual vector form (3.4)');

% Eq 3.5: Scaled Quadratic Form Identity
check_eq(3.5, alpha_e_alphap_C_alpha - alpha(1)*(m_val(1) - a'*alpha), 'Quadratic form product simplification (3.5)');

% Eq 3.6: Cramer''s rule for alpha_1
check_eq(3.6, alpha(1) - detE/detB, 'Cramer''s rule representation for alpha_1 (3.6)');

% Eq 3.7: Schur complement of D
check_eq(3.7,( m_val(1) - a'*alpha) - detD/detB, 'Schur complement matrix identity for D (3.7)');

% Eq 3.8: Main Section 3 Determinant Ratio Identity
check_eq(3.8, alpha_e_alphap_C_alpha*detB^2 - (detD*detE), 'Determinant ratio product formula (3.3 / 3.8)');

%% -------------------------------------------------------------------------
% SECTION 4 EQUATION CHECKS (Cauchy-Binet & Matrix M)
%--------------------------------------------------------------------------
% Eq 4.2: Schur complement of M
detM = det(M);   % to calculate once

check_eq(4.2, (sum(w(1:k)) - a'*(B\e)) - detM/detB, 'Schur complement matrix identity for M (4.2)');

% Eq 4.3: Second bracketed identity part 1
check_eq(4.3, (-sum(w(1:k)) + alpha'*e) - (-sum(w(1:k)) + a'*(B\e)), 'Moment replacement equivalence (4.3 pt.1)');

% Eq 4.3: Second bracketed identity part 2
check_eq(4.3, (-sum(w(1:k)) + alpha'*e) - (-detM/detB), 'Determinant equivalence for second bracket (4.3 pt.2)');

% Eq 4.4: Factorization check M = K*W*L''
K_mat = sym(zeros(k+1, n+1)); L_mat = sym(zeros(k+1, n+1));
K_mat(1,:) = 1; L_mat(1,:) = 1;
for r = 2:k+1
    K_mat(r, 2:end) = t.^(r);
    L_mat(r, 2:end) = t.^(r-1);
end
W_mat = sym(zeros(n+1));
W_mat(1,1) = -sum(w(k+1:n));
for i = 1:n
    W_mat(i+1, i+1) = w(i);
end
check_eq(4.4, M - K_mat*W_mat*L_mat', 'Matrix factorization M = KWL'' (4.4)');

% Table 1: Check that the formulas in Table 1 are correct
all_match = CBsubmatrix_dets(n,k,t,w);
check_eq('',sym(all_match - 1),'Determinant formulas in Table 1')

% Eq 4.5 and 4.6: Determinant of M is sum of all (4.5) and (4.6) terms
cb_sum = cauchybinetM(n,k,t,w);
check_eq(4.6,cb_sum - detM, 'Determinant of M is sum of all (4.5) and (4.6) terms')

% Eq 4.7, 4.8: Verify the comment following (4.8)
all_match = 1;  % running check on matches
if k == n
    all_match =1;  % There are no column 3 terms. So det(M) <= 0.
else
    B_sets = nchoosek(1:n,k+1); % all potential B sets
    % loop over all nchoose(n,k+1) sets
    for idx = 1:nchoosek(n,k+1)
        B_set = B_sets(idx,:);
        final_difference = check_4p7_4p8(n,k,B_set,t,w);
        B_set_match = isAlways( simplify(final_difference)== 0 );
        all_match = all_match*B_set_match;
    end
end
check_eq(4.8,sym(all_match - 1), 'Comment following (4.8) relating (4.7) to (4.8)')

% Eq.4.8, check that f(0) = 0%
all_match = 1;  % running check on matches
if k == n
    all_match =1;  % There are no column 3 terms. So det(M) <= 0.
else
    B_sets = nchoosek(1:n,k+1); % all potential B sets
    % loop over all nchoose(n,k+1) sets
    for idx = 1:nchoosek(n,k+1)
        B_set = B_sets(idx,:);
        f_of_zero = check_f_of_zero(n,k,B_set,t);
        B_set_match = isAlways( simplify(f_of_zero)== 0 );
        all_match = all_match * B_set_match;
    end
end
check_eq(4.8,sym(all_match - 1), 'Check that f(0)= 0 for f in (4.8)')

%
% Eq 4.9: Verify the derivative formula (4.9)
all_match = 1;  % running check on matches
if k == n
    all_match = 1;  % There are no column 3 terms. So det(M) <= 0.
else
    B_sets = nchoosek(1:n,k+1); % all potential B sets
    % loop over all nchoose(n,k+1) sets
    for idx = 1:nchoosek(n,k+1)
        B_set = B_sets(idx,:);
        final_difference = check_derivative(n,k,B_set,t);
        B_set_match = isAlways( simplify(final_difference)== 0 );
        all_match = all_match*B_set_match;
    end
end
check_eq(4.9,sym(all_match - 1), 'Check the derivative formula (4.9)')

%% ========================================================================
% Validation Helper Function
%==========================================================================
function check_eq(eq_num, expression, eq_label)
    % Simplify expression and handle multiple checks for absolute safety
    nsteps = 30;  % reduce for faster program, increase for more success
    simplified = simplify(expression,'steps',nsteps);
    if isequal(simplified, sym(zeros(size(simplified)))) || all(simplified(:) == 0)
        fprintf('Equation (%.1f) [PASS]: %s\n', eq_num, eq_label);
    else
        try
            if isAlways(simplified == 0)
                fprintf('Equation (%.1f) [PASS]: %s\n', eq_num, eq_label);
                return;
            end
        catch
        end
        fprintf('Equation (%.1f) [FAIL]: %s\n', eq_num, eq_label);
        %disp(simplified);    %causes program to freeze at times
        charsimplified = char(simplified);
        nprint = 200;  % number of characters to print
        l_charsimplified = length(charsimplified);
        if nprint < l_charsimplified
            charsimplified = [charsimplified(1:nprint),'...' ];
            disp(['Residual expression remaining (',num2str(nprint), ...
                ' of ',num2str(l_charsimplified),' characters):']);
        else
            disp('Residual expression remaining:');
        end
        disp( charsimplified )
    end
end

%% =========================================================================
% Function for comparing equations (4.7) and (4.8)
%==========================================================================
function final_difference = check_4p7_4p8(n,k,B_set,t,w)
% This function checks the comment following (4.8).
% It calculates the difference between the left hand side
% of (4.7) and f(t) in (4.8) times (prod of (t_i - t_j)^2 for i in set A 
% and j in set A for i < j) times (prod of w(i) for i in B). 
% Input: n - A is m by n
%        k - the subspace dimension in PCR and PLS
%        t - symbolic vector of length m
%        w - symbolic vector of length m
% 
syms s positive

%*******>>>  2. Define the Index Sets

%B_sets = nchoosek(1:n,k+1); % all potential B sets
%B = B_sets( randi( nchoosek(n+1,k+1) ), :);  % choose one set to test
B = B_set;
ell = max(B);  % the largest integer in B
A = B(1:k); % B without ell, works since nchoosek uses increasing order

% 1. Construct f(s)
% prod_A_t = product of t_i for i in A
prod_A_t = prod(t(A));
prod_A=prod_A_t;

% sum_A_prod_rest = sum of (product of t_i for i in A, except i ~= j)
term_sum = sym(0);
for j = A
    prod_rest = prod(t(setdiff(A, j)));
    term_sum = term_sum + prod_rest;
end

% f(s)
f(s) = (prod_A_t + s * term_sum) * prod(t(A) - s)^2 - prod_A_t^3;
t_l = t(ell);

f_tl = f(t_l);
%*******>>> 4. Build the LHS Expression (lhs4p7)

% --- Component 1 of lhs4p7 ---
sum_prod_B = sym(0);
for j = B
    B_except_j = setdiff(B, j);
    sum_prod_B = sum_prod_B + prod(t(B_except_j));
end

prod_diff_B = sym(1);
for idx1 = 1:numel(B)
    for idx2 = (idx1+1):numel(B)
        prod_diff_B = prod_diff_B * (t(B(idx1)) - t(B(idx2)))^2;
    end
end

term1_lhs = sum_prod_B * prod_diff_B * prod(w(B));

% --- Component 2 of lhs4p7 ---
prod_diff_A = sym(1);
for idx1 = 1:numel(A)
    for idx2 = (idx1+1):numel(A)
        prod_diff_A = prod_diff_A * (t(A(idx1)) - t(A(idx2)))^2;
    end
end

term2_lhs = (prod_A^3) * prod_diff_A * prod(w(A)) * w(ell);

% Combine to create lhs4p7

lhs4p7 = term1_lhs - term2_lhs;

%*******>>> 5. Verify the Identity

% Define the verification term to subtract
subtraction_term = f_tl * prod_diff_A * prod(w(B));

% Calculate the final difference
final_difference = lhs4p7 - subtraction_term;
end

%% ========================================================================
% Function to calculate the Cauchy-Binet sum for M using (4.5) and (4.6)
%==========================================================================
function cb_sum = cauchybinetM(n,k,t,w)
% This function calculates the Cauchy-Binet sum for M using (4.5) and (4.6)
% Input: n - A is m by n
%        k - the subspace dimension in PCR and PLS
%        t - symbolic vector of length m
%        w - symbolic vector of length m


% Calculate cbsum (Combinatorial summation checking)
% Generate all combinations for A and B

combA = nchoosek(1:n, k);
combB = nchoosek(1:n, k+1);

cbsum_val = sym(0);

% Sum over all n choose k combinations for cb2
for s = 1:size(combA, 1)
    idxA = combA(s, :);
    
    % Products for the A-subset
    prod_t3 = prod(t(idxA).^3);
    prod_w = prod(w(idxA));
    
    % Vandermonde squared for the A-subset
    diffs = sym(1);
    for i = 1:length(idxA)
        for j = i+1:length(idxA)
            diffs = diffs * (t(idxA(i)) - t(idxA(j)))^2;
        end
    end
    
    % Remaining terms for cb2
    sum_w_tail = sum(w(k+1:n));
    
    % Add to sum
    cb2 = -prod_t3 * diffs * prod_w * sum_w_tail;
    cbsum_val = cbsum_val + cb2;
end

% Sum over all n choose (k+1) combinations for cb3
for s = 1:size(combB, 1)
    idxB = combB(s, :);
    
    % Products for the B-subset
    prod_w = prod(w(idxB));
    
    % Vandermonde squared for the B-subset
    diffs = sym(1);
    for i = 1:length(idxB)
        for j = i+1:length(idxB)
            diffs = diffs * (t(idxB(i)) - t(idxB(j)))^2;
        end
    end
    
    % Sum of products over j with i != j for the B-subset
    inner_sum = sym(0);
    for j_idx = 1:length(idxB)
        % Product of all t_i where i != j
        t_subset = t(idxB);
        t_subset(j_idx) = [];
        inner_sum = inner_sum + prod(t_subset);
    end
    
    % Add to sum
    cb3 = inner_sum * diffs * prod_w;
    cbsum_val = cbsum_val + cb3;
end
cb_sum = cbsum_val;

end

%% ========================================================================
% Function to check the determinant formulas in Table 1
%==========================================================================
function all_match = CBsubmatrix_dets(n,k,t,w)
% This function calculates the determinant formulas in Table 1
%    and compares them with the determinants of the actual submatrices
% Input: n - A is m by n
%        k - the subspace dimension in PCR and PLS
%        t - symbolic vector of length m
%        w - symbolic vector of length m
% all_match is 1, if all determinants match, otherwise its 0

nsteps=100;  % steps in MATLAB's simplify routine
% Detailed Minor Verification Framework (Table 1 Cross-Check)
%fprintf('\n=== DETAILED CAUCHY-BINET MINOR FORMULAS CROSS-CHECK ===\n');
all_combinations = nchoosek(1:(n+1), k+1);

% Construct K and L matrices according formulas after (4.3)
K = sym(zeros(k+1, n+1));
L = sym(zeros(k+1, n+1));

% First columns of K and L
K(:, 1) = [1; zeros(k, 1)];
L(:, 1) = [1; zeros(k, 1)];

% Remaining columns for i = 1 to n
for i = 1:n
    % K column values
    K(1, i+1) = 1;
    for r = 2:k+1
        K(r, i+1) = t(i)^(r); % t_i^2, t_i^3, ..., t_i^{k+1}
    end
    
    % L column values
    L(1, i+1) = 1;
    for r = 2:k+1
        L(r, i+1) = t(i)^(r-1); % t_i^1, t_i^2, ..., t_i^k
    end
end

% Construct W matrix
W = sym(zeros(n+1, n+1));
W(1,1) = -sum(w(k+1:end)); % - \sum_{i=k+1}^n w_i
for i = 1:n
    W(i+1, i+1) = w(i);
end
% Group 1: Subsets including Column 1 (Equation 4.5 forms: n choose k terms)
%fprintf('\n--- Group 1: Subsets including Column 1 (Eq 4.5 Minors) ---\n');

all_match = 1;  % running flag to test all equations match
for idx = 1:size(all_combinations, 1)
    S = all_combinations(idx, :);
    if ismember(1, S)
        % Map back to the index in terms of the underlying parameters (i_1, i_2...)
        % S(1) is 1, so the actual vector indices chosen are S(2:end)-1
        actual_indices = S(2:end) - 1;
        A_idx = actual_indices;
        
        % Calculate actual minors

        det_K_sub = simplify(det(K(:, S)));
        % Calculate by formula in Table 1
            % 1. Calculate product of t_i^2 for i in A
                 term1 = prod(t(A_idx).^2);        
            % 2. Calculate product of (t_j - t_i) for i,j in A where i < j
            term2 = 1;
            for i_idx = 1:k
                for j_idx = i_idx + 1:k
                    i_val = t(A_idx(i_idx));
                    j_val = t(A_idx(j_idx));
                    term2 = term2 * (j_val - i_val);
                end
             end    
             % Total product for the current subset A
             det_K_sub_table1 = term1 * term2;
        check_difference = det_K_sub - det_K_sub_table1;
        check_det_K_sub = all( isAlways(simplify(check_difference,"Steps",nsteps) == 0) );
        all_match = all_match * check_det_K_sub;

        det_L_sub = simplify(det(L(:, S)));
        % Calculate by formula in Table 1
            % 1. Calculate product of t_i^2 for i in A
                 term1 = prod(t(A_idx));    
            % 2. Calculate product of (t_j - t_i) for i,j in A where i < j
            term2 = 1;
            for i_idx = 1:k
                for j_idx = i_idx + 1:k
                    i_val = t(A_idx(i_idx));
                    j_val = t(A_idx(j_idx));
                    term2 = term2 * (j_val - i_val);
                end
            end    
            % Total product for the current subset 
            det_L_sub_table1 = term1 * term2;
        check_difference = det_L_sub - det_L_sub_table1;
        check_det_L_sub = all( isAlways(simplify(check_difference,"Steps",nsteps) == 0) );
        all_match = all_match * check_det_L_sub;

            
        det_W_sub = simplify(prod(diag(W(S, S))));
        % Calculate by formula in Table 1
            % Calculate the second parenthetical sum: \sum_{j=k+1}^{n} w_j
              sum_term = sum(w(k+1:n));
            % Calculate the product of w_i for the indices in A
              prod_term = prod(w(A_idx));  
            % Calculate the final value and store it
            det_W_sub_table1 = -prod_term * sum_term;
         check_difference = det_W_sub - det_W_sub_table1;
         check_det_W_sub = all( isAlways(simplify(check_difference,"Steps",nsteps) == 0) );
         all_match = all_match * check_det_W_sub;

    end
end

% Group 2: Subsets excluding Column 1 (Equation 4.6 forms: n choose k+1 terms)
%fprintf('\n--- Group 2: Subsets excluding Column 1 (Eq 4.6 Minors) ---\n');
for idx = 1:size(all_combinations, 1)
    S = all_combinations(idx, :);
    if ~ismember(1, S)
        % Map back to the index in terms of underlying parameters
        actual_indices = S - 1;
        B=actual_indices;
        
        % Calculate actual minors
        det_K_sub = simplify(det(K(:, S)));
        % Calculate by formula in Table 1
            % --- 1. Calculate the summation block: Sum( Prod( t_i ) ) ---
            sum_term = 0;
            for j_idx = 1:length(B)
                j = B(j_idx);
                % Elements in B excluding the current j
                prod_indices = B(B ~= j);
                
                if isempty(prod_indices)
                    prod_val = 1;
                else
                    prod_val = prod(t(prod_indices));
                end
                sum_term = sum_term + prod_val;
            end    
            % --- 2. Calculate the product block: Product of (t_j - t_i) for i < j ---
            prod_term = 1;
            for i_idx = 1:length(B)
                for j_idx = i_idx + 1:length(B)
                    % B(i_idx) and B(j_idx) inherently satisfy i < j in subset indexing
                    prod_term = prod_term * (t(B(j_idx)) - t(B(i_idx)));
                end
            end   
            % --- 3. Final calculation for the subset ---
            det_K_sub_table1 = sum_term * prod_term;
        check_difference = det_K_sub - det_K_sub_table1;
        check_det_K_sub = all( isAlways(simplify(check_difference,"Steps",nsteps) == 0) );
        all_match = all_match * check_det_K_sub;

        det_L_sub = simplify(det(L(:, S)));
        % Calculate by formula in Table 1
            % --- 1. Calculate the product block: Product of (t_j - t_i) for i < j ---
            prod_term = 1;
            for i_idx = 1:length(B)
                for j_idx = i_idx + 1:length(B)
                    % B(i_idx) and B(j_idx) inherently satisfy i < j in subset indexing
                    prod_term = prod_term * (t(B(j_idx)) - t(B(i_idx)));
                end
            end
            det_L_sub_table1 = prod_term;
      check_difference = det_L_sub - det_L_sub_table1;
      check_det_L_sub = all( isAlways(simplify(check_difference,"Steps",nsteps) == 0) );
      all_match = all_match * check_det_L_sub;
      
      det_W_sub = simplify(prod(diag(W(S, S))));
      %idx_B = B_subsets(i, :);
      % Calculate by formula in Table 1
          % Calculate the product of w_i for the indices in B
          prod_term = prod(w(B));
          det_W_sub_table1 = prod_term;
      check_difference = det_W_sub - det_W_sub_table1;
      check_det_W_sub = all( isAlways(simplify(check_difference,"Steps",nsteps) == 0) );
      all_match = all_match * check_det_W_sub;
    end
end

end

%% ========================================================================
% Function to check f(0) = 0 in (4.8)
%==========================================================================
function final_difference = check_f_of_zero(n,k,B_set,t)
% This function checks f(0) = 0 with f defined in (4.8)
% Input: n - A is m by n (m not needed)
%        k - dimension of the subspaces in PCR and PLS
%        B_set - a set of k+1 integers selected from 1 to n
%        t - a symobolic vector of length n

syms r real  % use r since t and s are in use (via the arguments)

ell = max(B_set);
A_set = B_set(1:k); % works since matlab uses increasing order in sets

% 1. Construct f(r)
% prod_A_t = product of t_i for i in A_set
prod_A_t = prod(t(A_set));

% sum_A_prod_rest = sum of (product of t_i for i in A_set, except i ~= j)
term_sum = sym(0);
for j = A_set
    prod_rest = prod(t(setdiff(A_set, j)));
    term_sum = term_sum + prod_rest;
end

% f(r)
f(r) = (prod_A_t + r * term_sum) * prod(t(A_set) - r)^2 - prod_A_t^3;

final_difference = f(0);

end

%% ========================================================================
% Function to check the derivative formula (4.9)
%==========================================================================
function final_difference = check_derivative(n,k,B_set,t)
% This function checks the derivative formula (4.9).
% The function forms the function f(t), differentiates it 
%    using Matlab's diff routine, calculates the formula (4.9)
%    and returns the difference. In the special case that 
%    t_ell is zero final_difference is set to f(t_ell).
%    When t_ell is zero f(t_ell) == 0 suffices to prove (4,7).
% Input: n - A is m by n (m not needed)
%        k - dimension of the subspaces in PCR and PLS
%        B_set - a set of k+1 integers selected from 1 to n
%        t - a symobolic vector of length n

syms r real  % use r since t and s are in use (via the arguments)

ell = max(B_set);
A_set = B_set(1:k); % works since matlab uses increasing order in sets
%A_sets = nchoosek(1:n,k); % all potential B sets
%A = A_sets( randi( nchoosek(n,k) ), :)  % choose just one set to check
      % the calculations are the same for any set (with different indices)

% 1. Construct f(r)
% prod_A_t = product of t_i for i in A_set
prod_A_t = prod(t(A_set));

% sum_A_prod_rest = sum of (product of t_i for i in A_set, except i ~= j)
term_sum = sym(0);
for j = A_set
    prod_rest = prod(t(setdiff(A_set, j)));
    term_sum = term_sum + prod_rest;
end

% f(r)
f(r) = (prod_A_t + r * term_sum) * prod(t(A_set) - r)^2 - prod_A_t^3;

if isAlways( t(ell) == 0, Unknown="false" )
    % in this case it suffices in Lemma 4.2 to have f( t(ell) ) == 0
    final_difference = f( t(ell) );
else
    % 2. Construct formula 4.9 df/ds
    term1 = sum( (1 ./ t(A_set)) .* ((t(A_set) + r) ./ (t(A_set) - r)) );
    term2 = sum( 1 ./ (t(A_set) - r) );
    term3 = sum( 1 ./ t(A_set) );
    prod_diff_sq = prod(t(A_set) - r)^2;
    
    dfds_formula = (-1) * (term1 + 2*r * term2 * term3) * prod_diff_sq * prod_A_t;
    dfds_formula = simplify(dfds_formula);
    
    % 3. calculate diff(f,r) - dfds_st so that it can be checked == 0
    diff_f = diff(f(r), r);
    diff_f = simplify(diff_f);
    final_difference = simplify(diff_f - dfds_formula);
end

end

end