%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Welfare_output, UE_output, MEC_output, Bid_output, Price_output] = Static_Case_GSP_MEC(T,Nrun,R,J)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global N I K R_N RN_max f_C_min tau_min tau_max Gamma_max Gamma_min d_avg d_k lambda_s a_Bar q_c q_l Delta_t epsilon kappa rho VM_f VM_v VM_C VM_I

%------------------------------------------------------------------
% set-up simulation parameter values
%------------------------------------------------------------------
% T = 50;                      % no. of auction round/time slot
% Nrun = 100;                   % no. of iterations to average the results
N = 1;                      % no. of MEC application/processor at the orchestrator
I = 2;                      % no. of MEC servers/nodes 
K = J;
Delta_t = 60;               % duration of time slot 1 min = 60 sec
epsilon = 0.001;
%------------------------------------------------------------------
% Server parameters  
%------------------------------------------------------------------
f = [3.3; 3.5; 3.2; 3.2; 3.3]*1e9 ;         % computing frequency (GHz) of each server
W = [2; 2; 2; 1; 1];                        % no. of cores in each VM of type n 
C = 4e6*[32; 24; 24; 16; 16];               % computing power (MIPS) --> convert to MB/sec (1 instruction 4 byte)
%------------------------------------------------------------------
% App-specific parameters
%------------------------------------------------------------------
tau_max = 200e-3;               % task completion deadline
tau_min = 20e-3;                % minimum required time to complete a task
f_C_min = 3.2e9;                % minimum CPU freq. requirement
%------------------------------------------------------------------
% UE-specific parameters
%------------------------------------------------------------------
a_Bar = 0.0278*ones(J,1); 	% UEs' monetary budget convert $20/month --> $20/(30*24) = 0.0278 ($/VM-hr)  
q_c = 0.5;  q_l=0.5;         	% QoE parameter to adjust trade-off between offloading cost and service latency
%--------------------------------------------------------------------------
% VM's parameters
%--------------------------------------------------------------------------
v = zeros(I,N);   R_N = sum(R);       RN_max = max(R_N);  
VM_f = zeros(RN_max,N);     VM_C = zeros(RN_max,N);     VM_v = zeros(RN_max,N);     VM_I = zeros(RN_max,N);

kappa = 10*1e-24;               % effective switched capacitance 
rho=[0.0452; 0.0435; 0.0385; 0.0186; 0.0175];   % scaling parameter for $/VM-hr --> $/VM-sec rates
Gamma_max = 0.95;            Gamma_min = 0.01;
%--------------
for i = 1:I
    for n = 1:N
        v(i,n) = rho(i)*kappa*(f(i).^2)*W(i,n)*3600;        % unit price ($/VM-hour)for VM of type n at MEC server i           
    end
end
%--------------
for n = 1:N
    r=1;
    for i = 1:I
        for mr = 1:R(i,n)
            VM_f(r,n) = f(i)*W(i,n);
            VM_C(r,n) = W(i,n)*C(i,n);
            VM_v(r,n) = v(i,n);
            VM_I(r,n) = i;
            r = r +1;
        end
    end
end
% v = [0.0175 0.0162]

%--------------------------------------------------------------------------
% offloading data parameters
%--------------------------------------------------------------------------
d_avg = randi([10,40],1,J);       % average data size 10 to 100 MB
d_k = zeros(K,N,T,Nrun);            % offloading data size
%------------------------------------------------------------------


%--------------------------------------------------------------------------
% initializing variables
%-------------------------------------------------------------------------- 
for var_initialization=1
x_RBB = zeros(K,RN_max,N,T,Nrun);   theta_RBB = zeros(RN_max,N,T,Nrun); eta_RBB = zeros(RN_max,N,T,Nrun);   eta_rem_RBB = zeros(RN_max,N,T,Nrun); 
x_BB = zeros(K,RN_max,N,T,Nrun);    theta_BB = zeros(RN_max,N,T,Nrun);  eta_BB = zeros(RN_max,N,T,Nrun);    eta_rem_BB = zeros(RN_max,N,T,Nrun);
x_AB = zeros(K,RN_max,N,T,Nrun);    theta_AB = zeros(RN_max,N,T,Nrun);  eta_AB = zeros(RN_max,N,T,Nrun);    eta_rem_AB = zeros(RN_max,N,T,Nrun);
x_CB = zeros(K,RN_max,N,T,Nrun);    theta_CB = zeros(RN_max,N,T,Nrun);  eta_CB = zeros(RN_max,N,T,Nrun);    eta_rem_CB = zeros(RN_max,N,T,Nrun);
x_VCG = zeros(K,RN_max,N,T,Nrun);   theta_VCG = zeros(RN_max,N,T,Nrun); eta_VCG = zeros(RN_max,N,T,Nrun);   eta_rem_VCG = zeros(RN_max,N,T,Nrun);      
       
x_RBB_prev = zeros(K,RN_max,N);     eta_RBB_prev = zeros(RN_max,N);     theta_RBB_prev = zeros(RN_max,N);   p_RBB_prev = zeros(K,N);          
x_BB_prev = zeros(K,RN_max,N);      eta_BB_prev = zeros(RN_max,N);      theta_BB_prev = zeros(RN_max,N);    p_BB_prev = zeros(K,N); 
x_AB_prev = zeros(K,RN_max,N);      eta_AB_prev = zeros(RN_max,N);      theta_AB_prev = zeros(RN_max,N);    p_AB_prev = zeros(K,N);
x_CB_prev = zeros(K,RN_max,N);      eta_CB_prev = zeros(RN_max,N);      theta_CB_prev = zeros(RN_max,N);    p_CB_prev = zeros(K,N); 
x_VCG_prev = zeros(K,RN_max,N);     eta_VCG_prev =  zeros(RN_max,N);            
                      
p_RBB = zeros(K,N,T,Nrun);          pp_avg_RBB = zeros(K,N,T);          p_avg_RBB = zeros(1,T);
p_BB = zeros(K,N,T,Nrun);           pp_avg_BB = zeros(K,N,T);           p_avg_BB = zeros(1,T);
p_AB = zeros(K,N,T,Nrun);           pp_avg_AB = zeros(K,N,T);           p_avg_AB = zeros(1,T);
p_CB = zeros(K,N,T,Nrun);           pp_avg_CB = zeros(K,N,T);           p_avg_CB = zeros(1,T);
p_VCG = zeros(K,N,T,Nrun);          pp_avg_VCG = zeros(K,N,T);          p_avg_VCG = zeros(1,T);

VI = zeros(I,T,Nrun);               VI_avg = zeros(I,T);
b_RBB = zeros(RN_max,N,T,Nrun);     b_MEC_RBB = zeros(I,N,T,Nrun);      bMEC_avg_RBB = zeros(I,T); 
b_BB = zeros(RN_max,N,T,Nrun);      b_MEC_BB = zeros(I,N,T,Nrun);       bMEC_avg_BB = zeros(I,T);  
b_AB = zeros(RN_max,N,T,Nrun);      b_MEC_AB = zeros(I,N,T,Nrun);       bMEC_avg_AB = zeros(I,T);
b_CB = zeros(RN_max,N,T,Nrun);      b_MEC_CB = zeros(I,N,T,Nrun);       bMEC_avg_CB = zeros(I,T);
b_VCG = zeros(RN_max,N,T,Nrun);     b_MEC_VCG = zeros(I,N,T,Nrun);      bMEC_avg_VCG = zeros(I,T);
          
delta_RBB = zeros(K,N,T,Nrun);      ddelta_avg_RBB = zeros(K,N,T);      delta_avg_RBB = zeros(1,T);    
delta_BB = zeros(K,N,T,Nrun);       ddelta_avg_BB = zeros(K,N,T);       delta_avg_BB = zeros(1,T); 
delta_AB = zeros(K,N,T,Nrun);       ddelta_avg_AB = zeros(K,N,T);       delta_avg_AB = zeros(1,T); 
delta_CB = zeros(K,N,T,Nrun);       ddelta_avg_CB = zeros(K,N,T);       delta_avg_CB = zeros(1,T); 
delta_VCG = zeros(K,N,T,Nrun);      ddelta_avg_VCG = zeros(K,N,T);      delta_avg_VCG = zeros(1,T);

a_k_RBB = zeros(K,N,T,Nrun);        aa_k_avg_RBB = zeros(K,N,T);        a_k_avg_RBB = zeros(1,T);   
a_k_BB = zeros(K,N,T,Nrun);         aa_k_avg_BB = zeros(K,N,T);         a_k_avg_BB = zeros(1,T); 
a_k_AB = zeros(K,N,T,Nrun);         aa_k_avg_AB = zeros(K,N,T);         a_k_avg_AB = zeros(1,T);
a_k_CB = zeros(K,N,T,Nrun);         aa_k_avg_CB = zeros(K,N,T);         a_k_avg_CB = zeros(1,T);
a_k_VCG = zeros(K,N,T,Nrun);        aa_k_avg_VCG = zeros(K,N,T);        a_k_avg_VCG = zeros(1,T);	

Q_cost_RBB = zeros(J,T,Nrun);       QQ_cost_avg_RBB = zeros(J,T);       Q_cost_avg_RBB = zeros(1,T);  
Q_cost_BB = zeros(J,T,Nrun);        QQ_cost_avg_BB = zeros(J,T);        Q_cost_avg_BB = zeros(1,T); 
Q_cost_AB = zeros(J,T,Nrun);        QQ_cost_avg_AB = zeros(J,T);        Q_cost_avg_AB = zeros(1,T);
Q_cost_CB = zeros(J,T,Nrun);        QQ_cost_avg_CB = zeros(J,T);        Q_cost_avg_CB = zeros(1,T);
Q_cost_VCG = zeros(J,T,Nrun);       QQ_cost_avg_VCG = zeros(J,T);       Q_cost_avg_VCG = zeros(1,T);
         
Q_RBB = zeros(J,T,Nrun);            QQ_avg_RBB = zeros(J,T);            Q_avg_RBB = zeros(1,T);   
Q_BB = zeros(J,T,Nrun);             QQ_avg_BB = zeros(J,T);             Q_avg_BB = zeros(1,T);
Q_AB = zeros(J,T,Nrun);             QQ_avg_AB = zeros(J,T);             Q_avg_AB = zeros(1,T);
Q_CB = zeros(J,T,Nrun);             QQ_avg_CB = zeros(J,T);             Q_avg_CB = zeros(1,T);
Q_VCG = zeros(J,T,Nrun);            QQ_avg_VCG = zeros(J,T);            Q_avg_VCG = zeros(1,T);
                
u_MEC_RBB = zeros(I,T,Nrun);        uMEC_avg_RBB = zeros(I,T);  
u_MEC_BB = zeros(I,T,Nrun);         uMEC_avg_BB = zeros(I,T);
u_MEC_AB = zeros(I,T,Nrun);         uMEC_avg_AB = zeros(I,T);
u_MEC_CB = zeros(I,T,Nrun);         uMEC_avg_CB = zeros(I,T);
u_MEC_VCG = zeros(I,T,Nrun);        uMEC_avg_VCG = zeros(I,T);      

upm_RBB = zeros(I,N,T,Nrun);        ppm_RBB = zeros(I,N,T);     upm_avg_RBB = zeros(1,T); 
upm_BB = zeros(I,N,T,Nrun);         ppm_BB = zeros(I,N,T);      upm_avg_BB = zeros(1,T); 
upm_AB = zeros(I,N,T,Nrun);         ppm_AB = zeros(I,N,T);      upm_avg_AB = zeros(1,T); 
upm_CB = zeros(I,N,T,Nrun);         ppm_CB = zeros(I,N,T);      upm_avg_CB = zeros(1,T); 
upm_VCG = zeros(I,N,T,Nrun);        ppm_VCG = zeros(I,N,T);     upm_avg_VCG = zeros(1,T); 
                 
z_RBB = zeros(N,T,Nrun);            z_avg_RBB = zeros(1,T);   
z_BB = zeros(N,T,Nrun);             z_avg_BB = zeros(1,T);
z_AB = zeros(N,T,Nrun);             z_avg_AB = zeros(1,T);
z_CB = zeros(N,T,Nrun);             z_avg_CB = zeros(1,T);
z_VCG = zeros(N,T,Nrun);            z_avg_VCG = zeros(1,T);
                 
SW_RBB = zeros(N,T,Nrun);           SW_avg_RBB = zeros(1,T);  
SW_BB = zeros(N,T,Nrun);            SW_avg_BB = zeros(1,T);
SW_AB = zeros(N,T,Nrun);            SW_avg_AB = zeros(1,T);
SW_CB = zeros(N,T,Nrun);            SW_avg_CB = zeros(1,T);
SW_VCG = zeros(N,T,Nrun);           SW_avg_VCG = zeros(1,T);   

b_s = zeros(K,N,T,Nrun);            pos_b_avg = zeros(K,T);
u_s = zeros(K,N,T,Nrun);            pos_u_avg = zeros(K,T);
%--------------------------------------------------------------------------
end

%--------------------------------------------------------------------------
% updating task priority scores
%--------------------------------------------------------------------------
lambda_s = zeros(K,N);
for n = 1:N
    for s = 1:J
        lambda_s(s,n) = (d_avg(s)*1e6)./(tau_max(n));
    end
    lambda_s(:,N) = sort(lambda_s(:,n),'descend'); 
end
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Repeated Auction Begins (t=1~T)
%--------------------------------------------------------------------------
% K_init = J;          % no. of incoming tasks for initial auction round (t=1)
for t = 1:T
    for nr = 1:Nrun
        %----------------------------------------------------------------------
        % Task arrival
        %----------------------------------------------------------------------
        K_N = J;                    % assume same no. of UEs in every round
%          K_N = randi([1,J]);        % assume no. of incoming tasks = no. of UEs
         K_j = randperm(K_N,K_N);       % randomly choose which UE sends the incoming offloading requests

        for s = 1:K_N
            j_s = K_j(s);
%             d_k(j_s,n,t,nr) = poissrnd(d_avg(j_s))*1e6;     % data size (MB) follow Poisson distribution
            d_k(j_s,n,t,nr) = d_avg(j_s)*1e6;                % assume, fixed data size in every round
        end
        %----------------------------------------------------------------------
        
        %----------------------------------------------------------------------
        if(t==1)
            for n = 1:N
                theta_RBB_prev(:,n) = VM_f(:,n)./f_C_min(n);    theta_BB_prev(:,n) = VM_f(:,n)./f_C_min(n);     theta_AB_prev(:,n) = VM_f(:,n)./f_C_min(n);     theta_CB_prev(:,n) = VM_f(:,n)./f_C_min(n); 
                if(K_N<R(n))
                    x_M = randperm(R_N(n),K_N);                         % assume random allocation of slots to VMs for round t=1
                    K_limit = K_N;
                else
                    x_M = randperm(R_N(n),R_N(n));
                    K_limit = R_N(n);
                end
                for s = 1:K_limit
                    r = x_M(s);
                    x_RBB_prev(s,r,n) = 1;      x_BB_prev(s,r,n) = 1;   x_AB_prev(s,r,n) = 1;   x_CB_prev(s,r,n) = 1;   x_VCG_prev(s,r,n) = 1;            
                    eta_RBB_prev(r,n) = 0;      eta_BB_prev(r,n) = 0;   eta_AB_prev(r,n) = 0;   eta_CB_prev(r,n) = 0;   eta_VCG_prev(r,n) = 0;
                    p_RBB_prev(s,n) = VM_v(r,n); p_BB_prev(s,n) = VM_v(r,n);    p_AB_prev(s,n) = VM_v(r,n);    p_CB_prev(s,n) = VM_v(r,n);      
                end
            end                                                        
        else
            x_RBB_prev = x_RBB(:,:,:,t-1,nr);   x_BB_prev = x_RBB(:,:,:,t-1,nr);   x_AB_prev = x_RBB(:,:,:,t-1,nr);   x_CB_prev = x_RBB(:,:,:,t-1,nr);  x_VCG_prev = x_VCG(:,:,:,t-1,nr);       
            eta_RBB_prev = eta_RBB(:,:,t-1,nr);  eta_BB_prev = eta_RBB(:,:,t-1,nr);  eta_AB_prev = eta_RBB(:,:,t-1,nr);   eta_CB_prev = eta_RBB(:,:,t-1,nr);  eta_VCG_prev = eta_VCG(:,:,t-1,nr);
            
            theta_RBB_prev = theta_RBB(:,:,t-1,nr);  theta_BB_prev = theta_RBB(:,:,t-1,nr);   theta_AB_prev = theta_RBB(:,:,t-1,nr);    theta_CB_prev = theta_RBB(:,:,t-1,nr);
            p_RBB_prev = p_RBB(:,:,t-1,nr);          p_BB_prev = p_BB(:,:,t-1,nr);           p_AB_prev = p_RBB(:,:,t-1,nr);            p_CB_prev = p_RBB(:,:,t-1,nr);   
        end 
        %-------------------------------

        %------------------------------------------------------------------
        % update quality scores of VMs
        %------------------------------------------------------------------
        [theta_RBB(:,:,t,nr)] = Update_VMs_Quality_Scores(eta_RBB_prev);
        [theta_BB(:,:,t,nr)] = Update_VMs_Quality_Scores(eta_BB_prev);
        [theta_AB(:,:,t,nr)] = Update_VMs_Quality_Scores(eta_AB_prev);
        [theta_CB(:,:,t,nr)] = Update_VMs_Quality_Scores(eta_CB_prev);
        [theta_VCG(:,:,t,nr)] = Update_VMs_Quality_Scores(eta_VCG_prev);
        
        %------------------------------------------------------------------
        % collect bids from MEC servers
        %------------------------------------------------------------------
        for n = 1:N
            ri=0;
            for i = 1:I
                current_VM_RBB = zeros(R(i,n),1);   current_VM_AB = zeros(R(i,n),1);    current_VM_CB = zeros(R(i,n),1);
                  
                for r_s  = 1:R(i,n)
                    for s = 1:K
                        if(x_RBB_prev(s,r_s,n)==1)
                            current_VM_RBB(r_s) = s;
                        end
                        if(x_AB_prev(s,r_s,n)==1)
                            current_VM_AB(r_s) = s;
                        end
                        if(x_AB_prev(s,r_s,n)==1)
                            current_VM_CB(r_s) = s;
                        end
                    end 

                end

                [~,b_RBB(ri+1:ri+R(i,n),n,t,nr),~,~] = Bidding_strategy(K,R(i,n),lambda_s(:,n),VM_v(ri+1:ri+R(i,n),n),theta_RBB(ri+1:ri+R(i,n),n,t,nr),current_VM_RBB,theta_RBB_prev(ri+1:ri+R(i,n),n),p_RBB_prev(:,n));
                [b_BB(ri+1:ri+R(i,n),n,t,nr),~,~,~] = Bidding_strategy(K,R(i,n),lambda_s(:,n),VM_v(ri+1:ri+R(i,n),n),theta_BB(ri+1:ri+R(i,n),n,t,nr),current_VM_RBB,theta_BB_prev(ri+1:ri+R(i,n),n),p_BB_prev(:,n));
                [~,~,b_AB(ri+1:ri+R(i,n),n,t,nr),~] = Bidding_strategy(K,R(i,n),lambda_s(:,n),VM_v(ri+1:ri+R(i,n),n),theta_AB(ri+1:ri+R(i,n),n,t,nr),current_VM_AB,theta_AB_prev(ri+1:ri+R(i,n),n),p_AB_prev(:,n));
                [~,~,~,b_CB(ri+1:ri+R(i,n),n,t,nr)] = Bidding_strategy(K,R(i,n),lambda_s(:,n),VM_v(ri+1:ri+R(i,n),n),theta_CB(ri+1:ri+R(i,n),n,t,nr),current_VM_CB,theta_CB_prev(ri+1:ri+R(i,n),n),p_CB_prev(:,n));
                
                ri = ri+R(i,n);
            end
            
        end
    
        [x_RBB(:,:,:,t,nr),eta_RBB(:,:,t,nr),p_RBB(:,:,t,nr),upm_RBB(:,:,t,nr),u_MEC_RBB(:,t,nr),z_RBB(:,t,nr),eta_rem_RBB(:,:,t,nr),b_s(:,:,t,nr),u_s(:,:,t,nr)] = GSP_Mechanism(theta_RBB(:,:,t,nr),eta_RBB_prev,b_RBB(:,:,t,nr));
        [x_BB(:,:,:,t,nr),eta_BB(:,:,t,nr),p_BB(:,:,t,nr),upm_BB(:,:,t,nr),u_MEC_BB(:,t,nr),z_BB(:,t,nr),eta_rem_BB(:,:,t,nr),b_s(:,:,t,nr),u_s(:,:,t,nr)] = GSP_Mechanism(theta_BB(:,:,t,nr),eta_BB_prev,b_BB(:,:,t,nr));
        [x_AB(:,:,:,t,nr),eta_AB(:,:,t,nr),p_AB(:,:,t,nr),upm_AB(:,:,t,nr),u_MEC_AB(:,t,nr),z_AB(:,t,nr),eta_rem_AB(:,:,t,nr),b_s(:,:,t,nr),u_s(:,:,t,nr)] = GSP_Mechanism(theta_AB(:,:,t,nr),eta_AB_prev,b_AB(:,:,t,nr));
        [x_CB(:,:,:,t,nr),eta_CB(:,:,t,nr),p_CB(:,:,t,nr),upm_CB(:,:,t,nr),u_MEC_CB(:,t,nr),z_CB(:,t,nr),eta_rem_CB(:,:,t,nr),b_s(:,:,t,nr),u_s(:,:,t,nr)] = GSP_Mechanism(theta_CB(:,:,t,nr),eta_CB_prev,b_CB(:,:,t,nr));
        [x_VCG(:,:,:,t,nr),b_VCG(:,:,t,nr),p_VCG(:,:,t,nr),upm_VCG(:,:,t,nr),eta_VCG(:,:,t,nr),u_MEC_VCG(:,t,nr),z_VCG(:,t,nr),eta_rem_VCG(:,:,t,nr)] = Knapsack_VCG(R,theta_VCG(:,:,t,nr),eta_VCG_prev);
        
        for n = 1:N
            ri=0;
            for r = 1:R_N(n)
                r_i = VM_I(r,n);
                VI(r_i,t,nr)=VM_v(r,n);
            end
            for i = 1:I
                b_MEC_RBB(i,n,t,nr)=sum(b_RBB(ri+1:ri+R(i,n),n,t,nr))./R(i,n);
                b_MEC_BB(i,n,t,nr)=sum(b_BB(ri+1:ri+R(i,n),n,t,nr))./R(i,n);
                b_MEC_AB(i,n,t,nr)=sum(b_AB(ri+1:ri+R(i,n),n,t,nr))./R(i,n);
                b_MEC_CB(i,n,t,nr)=sum(b_CB(ri+1:ri+R(i,n),n,t,nr))./R(i,n);
                b_MEC_VCG(i,n,t,nr) = sum(b_VCG(ri+1:ri+R(i,n),n,t,nr))./R(i,n);
                ri = ri+R(i,n);
            end
        end
        
        %----------------------------------------------------------------------
        % UE QoE estimation
        %----------------------------------------------------------------------
        [delta_RBB(:,:,t,nr),a_k_RBB(:,:,t,nr),Q_cost_RBB(:,t,nr),Q_RBB(:,t,nr),SW_RBB(:,t,nr)] = UE_QoE(J,x_RBB(:,:,:,t,nr),p_RBB(:,:,t,nr),upm_RBB(:,:,t,nr),eta_rem_RBB(:,:,t,nr));
        [delta_BB(:,:,t,nr),a_k_BB(:,:,t,nr),Q_cost_BB(:,t,nr),Q_BB(:,t,nr),SW_BB(:,t,nr)] = UE_QoE(J,x_BB(:,:,:,t,nr),p_BB(:,:,t,nr),upm_BB(:,:,t,nr),eta_rem_BB(:,:,t,nr));
        [delta_AB(:,:,t,nr),a_k_AB(:,:,t,nr),Q_cost_AB(:,t,nr),Q_AB(:,t,nr),SW_AB(:,t,nr)] = UE_QoE(J,x_AB(:,:,:,t,nr),p_AB(:,:,t,nr),upm_AB(:,:,t,nr),eta_rem_AB(:,:,t,nr));
        [delta_CB(:,:,t,nr),a_k_CB(:,:,t,nr),Q_cost_CB(:,t,nr),Q_CB(:,t,nr),SW_CB(:,t,nr)] = UE_QoE(J,x_CB(:,:,:,t,nr),p_CB(:,:,t,nr),upm_CB(:,:,t,nr),eta_rem_CB(:,:,t,nr));
        [delta_VCG(:,:,t,nr),a_k_VCG(:,:,t,nr),Q_cost_VCG(:,t,nr),Q_VCG(:,t,nr),SW_VCG(:,t,nr)] = UE_QoE(J,x_VCG(:,:,:,t,nr),p_VCG(:,:,t,nr),upm_VCG(:,:,t,nr),eta_rem_VCG(:,:,t,nr));
    end

    x_RBB_avg = sum(mean(x_RBB(:,:,:,t,:),5));  x_BB_avg = sum(mean(x_RBB(:,:,:,t,:),5));   x_AB_avg = sum(mean(x_RBB(:,:,:,t,:),5));   x_CB_avg = sum(mean(x_RBB(:,:,:,t,:),5));
    x_VCG_avg = sum(mean(x_RBB(:,:,:,t,:),5));
    assigned_UEs_RBB = 0;   assigned_UEs_BB = 0;    assigned_UEs_AB = 0;    assigned_UEs_CB = 0;    assigned_UEs_VCG=0;
    for r = 1:R_N
        if(x_RBB_avg(r)==1)
            assigned_UEs_RBB = assigned_UEs_RBB + 1;
        end
        if(x_BB_avg(r)==1)
            assigned_UEs_BB = assigned_UEs_BB + 1;
        end
        if(x_AB_avg(r)==1)
            assigned_UEs_AB = assigned_UEs_AB + 1;
        end
        if(x_CB_avg(r)==1)
            assigned_UEs_CB = assigned_UEs_CB + 1;
        end
        if(x_VCG_avg(r)==1)
            assigned_UEs_VCG = assigned_UEs_VCG + 1;
        end
    end
    
    for output=1
    %----------------------------------------
    % Avg. bid of submitted by each server i
    %----------------------------------------
    VI_avg(:,t) = mean(VI(:,t,:),3);
    bMEC_avg_RBB(:,t) = mean(b_MEC_RBB(:,:,t,:),4);     bMEC_avg_BB(:,t) = mean(b_MEC_BB(:,:,t,:),4);      bMEC_avg_AB(:,t) = mean(b_MEC_AB(:,:,t,:),4);  bMEC_avg_CB(:,t) = mean(b_MEC_CB(:,:,t,:),4);
    bMEC_avg_VCG(:,t) = mean(b_MEC_VCG(:,:,t,:),4);

    %----------------------------------------
    % Avg. allocation prices in each round t
    %----------------------------------------
    pp_avg_RBB(:,:,t) = mean(p_RBB(:,:,t,:),4);     p_avg_RBB(:,t) = mean(pp_avg_RBB(:,:,t),1);
    pp_avg_BB(:,:,t) = mean(p_BB(:,:,t,:),4);       p_avg_BB(:,t) = mean(pp_avg_BB(:,:,t),1);
    pp_avg_BB(:,:,t) = mean(p_AB(:,:,t,:),4);       p_avg_AB(:,t) = mean(pp_avg_AB(:,:,t),1);
    pp_avg_CB(:,:,t) = mean(p_CB(:,:,t,:),4);       p_avg_CB(:,t) = mean(pp_avg_CB(:,:,t),1);
    pp_avg_VCG(:,:,t) = mean(p_VCG(:,:,t,:),4);     p_avg_VCG(:,t) = mean(pp_avg_VCG(:,:,t),1);

    %----------------------------------------
    % Avg. task execution latency in each round t
    %----------------------------------------
    ddelta_avg_RBB(:,:,t) = mean(delta_RBB(:,:,t,:),4);     delta_avg_RBB(:,t) = sum(ddelta_avg_RBB(:,:,t),1)./assigned_UEs_RBB;    % avg exection latency of each UE for GSP->RBB
    ddelta_avg_BB(:,:,t) = mean(delta_BB(:,:,t,:),4);       delta_avg_BB(:,t) = sum(ddelta_avg_BB(:,:,t),1)./assigned_UEs_BB;
    ddelta_avg_AB(:,:,t) = mean(delta_AB(:,:,t,:),4);       delta_avg_AB(:,t) = sum(ddelta_avg_AB(:,:,t),1)./assigned_UEs_AB;
    ddelta_avg_CB(:,:,t) = mean(delta_CB(:,:,t,:),4);       delta_avg_CB(:,t) = sum(ddelta_avg_CB(:,:,t),1)./assigned_UEs_CB;
    ddelta_avg_VCG(:,:,t) = mean(delta_VCG(:,:,t,:),4);     delta_avg_VCG(:,t) = sum(ddelta_avg_VCG(:,:,t),1)./assigned_UEs_VCG;     % avg exection latency of each UE for VCG    
    
    %----------------------------------------
    % Avg. offloading cost ($/task) in each round t
    %----------------------------------------
    aa_k_avg_RBB(:,:,t) = mean(a_k_RBB(:,:,t,:),4);     a_k_avg_RBB(:,t) = sum(aa_k_avg_RBB(:,:,t),1)./assigned_UEs_RBB;
    aa_k_avg_BB(:,:,t) = mean(a_k_BB(:,:,t,:),4);       a_k_avg_BB(:,t) = sum(aa_k_avg_BB(:,:,t),1)./assigned_UEs_BB;
    aa_k_avg_AB(:,:,t) = mean(a_k_AB(:,:,t,:),4);       a_k_avg_AB(:,t) = sum(aa_k_avg_AB(:,:,t),1)./assigned_UEs_AB;
    aa_k_avg_CB(:,:,t) = mean(a_k_CB(:,:,t,:),4);       a_k_avg_CB(:,t) = sum(aa_k_avg_CB(:,:,t),1)./assigned_UEs_CB;
    aa_k_avg_VCG(:,:,t) = mean(a_k_VCG(:,:,t,:),4);     a_k_avg_VCG(:,t) = sum(aa_k_avg_VCG(:,:,t),1)./assigned_UEs_VCG;

    %----------------------------------------
    % UEs' avg. budget cost savings ratio
    %----------------------------------------
    QQ_cost_avg_RBB(:,t) = mean(Q_cost_RBB(:,t,:),3);   Q_cost_avg_RBB(t) = sum(QQ_cost_avg_RBB(:,t),1)./assigned_UEs_RBB;
    QQ_cost_avg_BB(:,t) = mean(Q_cost_BB(:,t,:),3);     Q_cost_avg_BB(t) = sum(QQ_cost_avg_BB(:,t),1)./assigned_UEs_BB;
    QQ_cost_avg_AB(:,t) = mean(Q_cost_AB(:,t,:),3);     Q_cost_avg_AB(t) = sum(QQ_cost_avg_AB(:,t),1)./assigned_UEs_AB;
    QQ_cost_avg_CB(:,t) = mean(Q_cost_CB(:,t,:),3);     Q_cost_avg_CB(t) = sum(QQ_cost_avg_CB(:,t),1)./assigned_UEs_CB;
    QQ_cost_avg_VCG(:,t) = mean(Q_cost_VCG(:,t,:),3);   Q_cost_avg_VCG(t) = sum(QQ_cost_avg_VCG(:,t),1)./assigned_UEs_VCG;

    %----------------------------------------
    % Avg. utility gain (%) of UEs in each round t
    %----------------------------------------
    QQ_avg_RBB(:,t) = mean(Q_RBB(:,t,:),3);             Q_avg_RBB(t) = sum(QQ_avg_RBB(:,t),1)./assigned_UEs_RBB;
    QQ_avg_BB(:,t) = mean(Q_BB(:,t,:),3);               Q_avg_BB(t) = sum(QQ_avg_BB(:,t),1)./assigned_UEs_BB;
    QQ_avg_AB(:,t) = mean(Q_AB(:,t,:),3);               Q_avg_AB(t) = sum(QQ_avg_AB(:,t),1)./assigned_UEs_AB;
    QQ_avg_CB(:,t) = mean(Q_CB(:,t,:),3);               Q_avg_CB(t) = sum(QQ_avg_CB(:,t),1)./assigned_UEs_CB;
    QQ_avg_VCG(:,t) = mean(Q_VCG(:,t,:),3);             Q_avg_VCG(t) = sum(QQ_avg_VCG(:,t),1)./assigned_UEs_VCG;

    %----------------------------------------
    % Avg. profit of each server i
    %----------------------------------------
    uMEC_avg_RBB(:,t) = mean(u_MEC_RBB(:,t,:),3);   uMEC_avg_BB(:,t) = mean(u_MEC_BB(:,t,:),3); uMEC_avg_AB(:,t) = mean(u_MEC_AB(:,t,:),3);     uMEC_avg_CB(:,t) = mean(u_MEC_CB(:,t,:),3);
    uMEC_avg_VCG(:,t) = mean(u_MEC_VCG(:,t,:),3);
    
    %----------------------------------------
    % Avg. profit margin (%) of servers
    %----------------------------------------
    ppm_RBB(:,:,t) = mean(upm_RBB(:,:,t,:),4);      upm_avg_RBB(:,t) = mean(ppm_RBB(:,:,t),1);
    ppm_BB(:,:,t) = mean(upm_BB(:,:,t,:),4);        upm_avg_BB(:,t) = mean(ppm_BB(:,:,t),1);
    ppm_AB(:,:,t) = mean(upm_AB(:,:,t,:),4);        upm_avg_AB(:,t) = mean(ppm_AB(:,:,t),1);
    ppm_CB(:,:,t) = mean(upm_CB(:,:,t,:),4);        upm_avg_CB(:,t) = mean(ppm_CB(:,:,t),1);
    ppm_VCG(:,:,t) = mean(upm_VCG(:,:,t,:),4);      upm_avg_VCG(:,t) = mean(ppm_VCG(:,:,t),1);
    %----------------------------------------
    % Avg. auction revenue in each round t
    %----------------------------------------
    z_avg_RBB(:,t) = mean(z_RBB(:,t,:),3);  z_avg_BB(:,t) = mean(z_BB(:,t,:),3);    z_avg_AB(:,t) = mean(z_AB(:,t,:),3);    z_avg_CB(:,t) = mean(z_CB(:,t,:),3);
    z_avg_VCG(:,t) = mean(z_VCG(:,t,:),3);
    
    %----------------------------------------
    % Avg. social welfare
    %----------------------------------------
    SW_avg_RBB(:,t) = mean(SW_RBB(:,t,:),3);    SW_avg_BB(:,t) = mean(SW_BB(:,t,:),3);  SW_avg_AB(:,t) = mean(SW_AB(:,t,:),3);  SW_avg_CB(:,t) = mean(SW_CB(:,t,:),3);
    SW_avg_VCG(:,t) = mean(SW_VCG(:,t,:),3);
    end
    %----------------------------------------------------------------------
    %----------------------------------------
    % Avg. bid and utility for each allocated position
    %----------------------------------------
    pos_b_avg = mean(b_s(:,:,t,:),4);
    pos_u_avg = mean(u_s(:,:,t,:),4);
t
end

Welfare_output = [SW_avg_RBB; SW_avg_BB; SW_avg_AB; SW_avg_CB; SW_avg_VCG; 
                  z_avg_RBB; z_avg_BB; z_avg_AB; z_avg_CB; z_avg_VCG; 
                  upm_avg_RBB; upm_avg_BB; upm_avg_AB; upm_avg_CB; upm_avg_VCG];
UE_output = [Q_avg_RBB; Q_avg_BB; Q_avg_AB; Q_avg_CB; Q_avg_VCG; 
             Q_cost_avg_RBB; Q_cost_avg_BB; Q_cost_avg_AB; Q_cost_avg_CB; Q_cost_avg_VCG; 
             delta_avg_RBB; delta_avg_BB; delta_avg_AB; delta_avg_CB; delta_avg_VCG; 
             a_k_avg_RBB; a_k_avg_BB; a_k_avg_AB; a_k_avg_CB; a_k_avg_VCG];
MEC_output = [uMEC_avg_RBB; uMEC_avg_BB; uMEC_avg_AB; uMEC_avg_CB; uMEC_avg_VCG];
Bid_output = [VI_avg; bMEC_avg_RBB; bMEC_avg_BB; bMEC_avg_AB; bMEC_avg_CB; bMEC_avg_VCG];
Price_output = [p_avg_RBB; p_avg_BB; p_avg_AB; p_avg_CB; p_avg_VCG];
Position_output = [pos_b_avg; pos_u_avg];

end % end of main function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [theta] = Update_VMs_Quality_Scores(eta_prev)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global N R_N RN_max VM_f VM_C f_C_min Delta_t Gamma_max Gamma_min
theta = zeros(RN_max,N);    Gamma = zeros(RN_max,N);    phi = zeros(RN_max,N);
%-------------------
for n = 1:N
    for r = 1:R_N(n)
        if(VM_C(r,n)~=0)
            Gamma(r,n) = eta_prev(r,n)./(VM_C(r,n)*Delta_t); % load capacity (BPC)of each VM of type n at MEC server i
        end 

        if(Gamma(r,n)>=Gamma_max)
            phi(r,n) = 0;
        elseif((Gamma(r,n)>Gamma_min)&&(Gamma(r,n)<Gamma_max))
            phi(r,n) = abs(Gamma(r,n)-Gamma_max)./(Gamma_max);
        else
            phi(r,n) = 1;
        end

        theta(r,n) = (VM_f(r,n)*phi(r,n))./f_C_min(n)+1; 
    end  
end
%-------------------
end         % end of function        
        



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,eta,p,upm_MEC,u_MEC_RBB,z_RBB,eta_rem_RBB,b_s,u_s] = GSP_Mechanism(theta,eta_prev,b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global I K N R_N RN_max Delta_t epsilon d_k lambda_s VM_C VM_v VM_I
x = zeros(K,RN_max,N);              eta = zeros(RN_max,N);  
p= zeros(K,N);                      u = zeros(RN_max,N);
b_s = zeros(K,N);                   u_s = zeros(K,N);
upm_MEC = zeros(I,N);               u_RBB=zeros(N,1);
pm = zeros(RN_max,N);               z_V = zeros(K,N);       
z_RBB = zeros(N,1);                 u_MEC_RBB = zeros(I,1);
eta_rem_RBB=zeros(RN_max,N);


%----------------------------------------------------------------------
% rank VMs
%----------------------------------------------------------------------
y =zeros(RN_max,N); theta_diff = zeros(RN_max,N);
for n = 1:N
    theta_sort = sort(theta(:,n),'descend');
    for r_s = 1:R_N(n)
        y(r_s,n) = theta(r_s,n)./b(r_s,n);
        if(r_s<R_N(n))
            theta_diff(r_s,n) = theta_sort(r_s,n)./theta_sort(r_s+1,n);
        else
            theta_diff(r_s,n) = 1;
        end
    end
    theta_diff_sort = sort(theta_diff(:,n),'descend');
    [~,R_indx] = sort(y(:,n),'descend');
    %----------------------------------------------------------------------
    % GSP Allocation/matching decisions
    %----------------------------------------------------------------------
    for s = 1:K        
        if(s<R_N)
            r_s = R_indx(s,n);  r_next = R_indx(s+1,n); r_i = VM_I(r_s,n);
            x(s,r_s,n) = 1;
            
            eta_rem_RBB(r_s,n) = max(eta_prev(r_s,n) - (VM_C(r_s,n)*Delta_t),0);
            eta(r_s,n) = eta_rem_RBB(r_s,n) + d_k(s,n);

%             p(s,n) = max(((theta(r_s,n)*b(r_next,n))./theta(r_next,n)),b(r_s,n)+epsilon);
            p(s,n) = theta_diff_sort(s)*b(r_next,n);
            
            u(r_s,n) = lambda_s(s,n)*theta(r_s,n)*(p(s,n) - VM_v(r_s,n));
            u_s(s,n) = u(r_s,n);
            b_s(s,n) = b(r_s,n);
            u_MEC_RBB(r_i) = u_MEC_RBB(r_i) + u(r_s,n);
            
            pm(r_s,n) = ((p(s,n) - VM_v(r_s,n))./p(s,n));
            upm_MEC(r_i,n) = upm_MEC(r_i,n) + pm(r_s,n);
            
            z_V(s,n) = lambda_s(s,n)*theta(r_s,n)*p(s,n);
        elseif(s==R_N)
            r_s = R_indx(s,n);  r_i = VM_I(r_s,n);
            x(s,r_s,n) = 1;
            
            eta_rem_RBB(r_s,n) = max(eta_prev(r_s,n) - (VM_C(r_s,n)*Delta_t),0);
            eta(r_s,n)= eta_rem_RBB(r_s,n) + d_k(s,n);
            
            p(s,n) = b(r_s,n) + epsilon;
            
            u(r_s,n) = lambda_s(s,n)*theta(r_s,n)*(p(s,n) - VM_v(r_s,n));
            u_s(s,n) = u(r_s,n);
            b_s(s,n) = b(r_s,n);
            u_MEC_RBB(r_i) = u_MEC_RBB(r_i) + u(r_s,n);
            
            pm(r_s,n) = ((p(s,n) - VM_v(r_s,n))./p(s,n));
            upm_MEC(r_i,n) = upm_MEC(r_i,n) + pm(r_s,n);
            
            z_V(s,n) = lambda_s(s,n)*theta(r_s,n)*p(s,n);       
        end
    end

    u_RBB(n) = mean(u(:,n));
    z_RBB(n) = sum(z_V(:,n));
end

end         % end of function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_VCG,b_VCG,p_VCG,upm_MEC,eta_VCG,u_MEC_VCG,z_VCG,eta_rem_VCG] = Knapsack_VCG(R,theta,eta_prev)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global I K N R_N RN_max Delta_t d_k lambda_s VM_v VM_C
% n = no. of items (bidders)
% m = no. of knapsacks (sellers)
% p = profit/valuation if item j
% w = weight of item j
% c = capacity of knapsack(s)

X_DP = zeros(RN_max,N);     % dynamic programming allocation
x_VCG = zeros(K,RN_max,N);          p_VCG = zeros(K,N);
eta_VCG = zeros(RN_max,N);          eta_rem_VCG = zeros(RN_max,N);

u_V = zeros(RN_max,N);              u_VCG = zeros(N,1);     u_MEC_VCG = zeros(I,1);
z_V = zeros(K,N);                   z_VCG = zeros(N,1); 
pm = zeros(RN_max,N);               upm_MEC = zeros(I,N);

b_VCG = VM_v;       % truthful bidding as in VCG
    
for n = 1:N
%----------------------------------------------------------------------
% Dynamic Programming solution from knapsack
%----------------------------------------------------------------------
    w=ones(RN_max,1);   z = zeros(1,K+1);       A = zeros(RN_max,K);        c=K;  % initialization
%     b_VCG(:,n) = VM_v(:,n);
    y=theta(:,n)./b_VCG(:,n);       % weight valuation of the item (i.e., VM)
    %------------------
    for j = 1:R_N(n)
        for d = K:-1:w(j)
            if ((z(d-w(j)+1) + y(j)) > z(d+1))
                z(d+1) = z(d-w(j)+1) + y(j);
            end

            if((z(d-w(j)+1) + y(j)) == z(d+1))
                A(j,d) = 1;
            else
                A(j,d) = 0;
            end
        end
    end
    %------------------
    %z_D = z(c+1);      % optimal solution valueA
    
    C = c;      count = 1;      opp_y = y;
    %------------------
    for j = R_N(n):-1:1
        if (C > 0)
            if(A(j,C) == 1)
                X_DP(j,n) = 1;
                opp_y(j) = 0;
                eta_rem_VCG(j,n) = max(eta_prev(j,n) - (VM_C(j,n)*Delta_t),0);
                eta_VCG(j,n) = eta_rem_VCG(j,n) + d_k(C,n);

                if(~isempty(max(opp_y(opp_y>0))))
                      [~, jj] = max(opp_y(opp_y>0));
                    p_VCG(C,n) = b_VCG(jj,n);
                else
                    p_VCG(C,n) = b_VCG(j,n);
                end

                u_V(j,n) = lambda_s(C,n)*theta(j,n)*(p_VCG(C,n) - VM_v(j,n));
                z_V(C,n) = lambda_s(C,n)*theta(j,n)*p_VCG(C,n);
                pm(j,n) = ((p_VCG(C,n) - VM_v(j,n))./p_VCG(C,n));

                C = C - w(j);
                count = count + 1;
            end
        end
    end
end
%--------------------------------------------
for n = 1:N
    % update VCG allocation decisions
    for s = 1:K
        for r = 1:R_N(n)
            if(X_DP(r,n)==1)
                x_VCG(s,r,n) = 1;
            end
        end
    end
    ri=0;
    for i = 1:I        
        upm_MEC(i,n) = sum(pm(ri+1:ri+R(i,n),n))./R(i,n);
        u_MEC_VCG(i) = sum(u_V(ri+1:ri+R(i,n),n))./R(i,n);
        ri = ri+R(i,n);
    end
    
    u_VCG(n) = mean(u_V(:,n));
    z_VCG(n) = sum(z_V(:,n));
end

end         % end of function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [delta,a_k,Q_cost,Q,SW] = UE_QoE(J,x,p,upm_MEC,eta_rem)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global I K N R_N d_k VM_C a_Bar tau_max tau_min q_c q_l
SW = zeros(1,N);                a_k = zeros(K,N);
Q_latency = zeros(J,1);         Q_cost = zeros(J,1);  	Q = zeros(J,1);

% p = p./3600;        % convert $/VM-hour to $/sec


[r_u,MEC_node] = channel_model(I,J);      % users' wireless channel gain estimation 

%----------------------------------------------------------------------
% Determine task execution latency 
%----------------------------------------------------------------------
delta_up = zeros(K,N);   delta_comp = zeros(K,N);    delta_wait = zeros(K,N); delta = zeros(K,N);
for n = 1:N
    for s = 1:K
        for r  = 1:R_N(n)
            if((s<=J)&&(x(s,r,n)==1))
                j = s;
                ii = MEC_node(j);

                % users' offloading QoE in terms of delay estimation
                delta_up(s,n) = d_k(s,n)./r_u(ii,j);
                delta_comp(s,n) = d_k(s,n)./VM_C(r,n);  
            end
        end
        for r=1:R_N(n)
            if((eta_rem(r,n)>=0) && (x(s,r,n)==1))
                    
                    delta_wait(s,n) = eta_rem(r,n)./VM_C(r,n);
            end
        end
    end
    %------------------------------
    for s = 1:K             
        delta(s,n) = delta_up(s) + delta_comp(s) + delta_wait(s,n);     % in sec/task
        a_k(s,n) = (delta_comp(s,n)./3600)*p(s,n);       % convert delta(VM-sec/task) --> delta(VM-hr/task), then (VM-hr/task)*($/VM-hr)=$/task
        if(s<=J)
            j=s;
            Q_cost(j) = (a_Bar(j)-a_k(s,n))./a_Bar(j);         % budget cost savings ratio
        end
        
       if(delta(s,n)<=tau_min(n)) 
           Q_latency(j) = Q_latency(j) + 1;
       elseif(delta(s,n)<=tau_max(n))
           Q_latency(j) = Q_latency(j) + (tau_max(n) - delta(s,n))./(tau_max(n)-tau_min(n));
       else
           Q_latency(j) = Q_latency(j) + 0;
       end
       
       % users' offloading utility 
       Q(j) = q_c*Q_cost(j) + q_l*(Q_latency(j)./2);
    end
    %------------------------------

    SW(n) = sum(Q) + sum(upm_MEC);
end           

end         % end of function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gamma,MEC] =  channel_model(I,J)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dist = zeros(I,J);                  % distances between users and MEC servers
MEC = zeros(1,J);                   % associated MEC node
h = zeros(I,J);                     % path loss between user and associated MEC node
gamma = zeros(I,J);                 % Uplink data rate (bps)

% path loss parameters for urban high-rise, urban low-rise, suburban LoS
mu_d = 2.12;        mu_0 = 29.2;        mu_f = 2.11;                            

%---------------------------------------------------
% random user and MEC node distribJtion
%---------------------------------------------------
area = 250;     % indoor area (meter-square)

U_x = -area + (area+area)*rand(J,1);
U_y = -area + (area+area)*rand(J,1);

m_x = min(U_x) + (2*max(abs(U_x)))*rand(I,1);
m_y = min(U_y) + (2*max(abs(U_y)))*rand(I,1);


%---------------------------------------------------
% radio propagation model
%---------------------------------------------------
f = 5.8;            % transmission frequency (assuming IEEE 802.11ac) --> 5.8GHz
BW = 80*1e6;        % channel bandwidth (assuming IEEE 802.11ac) 802.11ac--> 80MHz

P_u = 0.1;                                       % User's transmit power (in watt) 20 dBm (assuming smart device)
% P_m = 10;                                      % edge node's transmit power assJme, 40 dBm(50 dBm for drones, 36dBm for IoT gateway)
sigma_N = 1e-13;                                 % noise power -100 dBm/Hz --> 3.98 watt

for j = 1:J
    [U_x_mesh, m_x_mesh] = meshgrid(U_x(j),m_x);  [U_y_J_mesh, m_y_mesh] = meshgrid(U_y(j),m_y);
    dist(:,j) = sqrt((U_x_mesh-m_x_mesh).^2 + (U_y_J_mesh-m_y_mesh).^2);
    
    [~,MEC(j)] = min(dist(:,j));                        % User associates with the nearest MEC server
    i = MEC(j);
    
    L_dB = 10*mu_d*log10(dist(i,j)) + mu_0 + 10*mu_f*log10(f);      % in dB

    h(i,j) = 10.^(L_dB/10);                               % convert dB to ratio
    
    SNR = (P_u*h(i,j))./sigma_N;

    gamma(i,j) = (BW*log2(1 + SNR))./8;                   % in Byte/sec
end

end             % end of function




