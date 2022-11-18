%%
clc; clear all; close all;
tic % measuring start time



%% Run Main function and plot graphs
T = 50;
Nrun = 1;
J = 150;    
R = [80; 80];         % case-3

%--------------------------------------------------------------------------
count = 0;
[~, ~, ~, Bid_output_1, Price_output_1] = Static_Case_GSP_MEC(T,Nrun,R,J);
count = count + 1
[~, ~, ~, Bid_output_2, Price_output_2] = Static_Case_GSP_MEC(T,Nrun,R_2,J);
count = count + 1

toc 


%----------------------------------------------------------
%% Plot Graphs
%----------------------------------------------------------


%% 
figure(1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VI_1 = Bid_output_1(1:2,:) ;  VI_2 = Bid_output_2(1:2,:);       VI_3 = Bid_output_1(1:2,:);
b_RBB_1 = Bid_output_1(3:4,:);      b_VCG_1 = Bid_output_1(11:12,:);
b_RBB_2 = Bid_output_2(3:4,:);      b_VCG_2 = Bid_output_2(11:12,:);
b_RBB_3 = Bid_output_3(3:4,:);      b_VCG_3 = Bid_output_3(11:12,:);

p_RBB_1 = Price_output_1(1,:);      p_VCG_1 = Price_output_1(5,:);      
p_RBB_2 = Price_output_2(1,:);      p_VCG_2 = Price_output_2(5,:);   
p_RBB_3 = Price_output_3(1,:);      p_VCG_3 = Price_output_3(5,:);
    
%-----------------------------------------------
subplot(1,2,1);
%-----------------------------------------------
plot(1:T,VI_1(1,:),'k-', 'LineWidth', 1.5); hold on; grid on;
plot(1:T,b_RBB_1(1,:),'b-', 'LineWidth', 1.5); hold on;
plot(1:T,b_RBB_2(1,:),'r-', 'LineWidth', 1.5); hold on;
plot(1:T,b_RBB_3(1,:),'m-', 'LineWidth', 1.5); hold on;
plot(1:T,VI_1(2,:),'k:', 'LineWidth', 2); hold on;
plot(1:T,b_RBB_1(2,:),'b:', 'LineWidth', 2); hold on;
plot(1:T,b_RBB_2(2,:),'r:', 'LineWidth', 2); hold on;
plot(1:T,b_RBB_3(2,:),'m:', 'LineWidth', 2); hold on;
legend({'v_1','b_1, R=[150,1]','b_1, R=[1,150]','b_1, R=[80,80]','v_2','b_2, R=[150,1]','b_2, R=[1,150]','b_2, R=[80,80]'},'Location','northwest','NumColumns',2);
xlabel('Auction rounds, t');
ylabel('Bids ($/VM-hr)');
xlim([1,T]);
%-----------------------------------------------
subplot(1,2,2);
%-----------------------------------------------
plot(1:T,p_VCG_1,'b:', 'LineWidth', 2); hold on; grid on;
plot(1:T,p_VCG_2,'r:', 'LineWidth', 2); hold on;
plot(1:T,p_VCG_3,'m:', 'LineWidth', 2); hold on;
plot(1:T,p_RBB_1,'b-', 'LineWidth', 1.5); hold on; 
plot(1:T,p_RBB_2,'r-', 'LineWidth', 1.5); hold on;
plot(1:T,p_RBB_3,'m-', 'LineWidth', 1.5); hold on;
legend('VCG, R=[150,1]','VCG, R=[1,150]','VCG, R=[80,80]','Proposed GSP, R=[150,1]','Proposed GSP, R=[1,150]','Proposed GSP, R=[80,80]');
xlabel('Auction rounds, t');
ylabel('Allocation price ($/VM-hr)');
xlim([1,T]);
