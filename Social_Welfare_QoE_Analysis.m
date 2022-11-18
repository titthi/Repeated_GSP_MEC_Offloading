%%
clc; clear all; close all;
tic % measuring start time



%% Run Main function and plot graphs
T = 1;
Nrun = 1;
J = 50:50:550;
J_count = length(J);
K=300;
I=3;
R_1 = [100; 100; 100;];         % case-1
R_2 = [150; 150; 150];         % case-2 (assuming I=3 servers)
d_min = [5; 10; 20];
d_max = [20; 40; 100];
%--------------------------------------------------------------------------
Welfare_output_1 = zeros(15,J_count);    UE_output_1 = zeros(20,J_count);
Welfare_output_2 = zeros(15,J_count);    UE_output_2 = zeros(20,J_count);
Welfare_output_3 = zeros(15,J_count);    UE_output_3 = zeros(20,J_count);
Welfare_output_4 = zeros(15,J_count);    UE_output_4 = zeros(20,J_count);
Welfare_output_5 = zeros(15,J_count);    UE_output_5 = zeros(20,J_count);

for count=1:J_count
[Welfare_output_1(:,count), UE_output_1(:,count),~, ~, ~] = Dynamic_Case_GSP_MEC(T,Nrun,I,R_1,J(count),K,d_min(2),d_max(2));
[Welfare_output_2(:,count), UE_output_2(:,count),~, ~, ~] = Dynamic_Case_GSP_MEC(T,Nrun,I,R_2,J(count),K,d_min(2),d_max(2));
count
end
toc 


%----------------------------------------------------------
%% Plot Graphs
%----------------------------------------------------------
SW_RBB_1 = Welfare_output_1(1,:);   SW_VCG_1 = Welfare_output_1(5,:);
SW_RBB_2 = Welfare_output_2(1,:);   SW_VCG_2 = Welfare_output_2(5,:);
Q_RBB_1 = UE_output_1(1,:);    Q_VCG_1 = UE_output_1(5,:);
Q_RBB_2 = UE_output_2(1,:);    Q_VCG_2 = UE_output_2(5,:);
%% 
figure(1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(1,2,1);
%-----------------------------------------------
plot(J,SW_RBB_1,'b-', 'LineWidth', 1.5); hold on; grid on;
plot(J,SW_VCG_1,'b--', 'LineWidth', 2); hold on;
plot(J,SW_RBB_2,'r-', 'LineWidth', 1.5); hold on;
plot(J,SW_VCG_2,'r--', 'LineWidth', 2); hold on;
legend('GSP (R=300)','VCG(R=300)','GSP(R=450)','VCG (R=450)');
xlabel('No. of UEs, J');
ylabel('Social welfare');
xlim([min(J),max(J)]);
%-----------------------------------------------
subplot(1,2,2);
%-----------------------------------------------
plot(J,Q_RBB_1,'b-', 'LineWidth', 1.5); hold on; grid on;
plot(J,Q_VCG_1,'b--', 'LineWidth', 2); hold on;
plot(J,Q_RBB_2,'r-', 'LineWidth', 1.5); hold on;
plot(J,Q_VCG_2,'r--', 'LineWidth', 2); hold on;
legend('GSP (R=300)','VCG(R=300)','GSP(R=450)','VCG (R=450)');
xlabel('No. of UEs, J');
ylabel('Avg. UE utility gain');
xlim([min(J),max(J)]);
% 
% 
K_new=400;
for count=1:J_count
[Welfare_output_3(:,count), UE_output_3(:,count),~, ~, ~] = Dynamic_Case_GSP_MEC(T,Nrun,I,R_2,J(count),K_new,d_min(1),d_max(1));
[Welfare_output_4(:,count), UE_output_4(:,count),~, ~, ~] = Dynamic_Case_GSP_MEC(T,Nrun,I,R_2,J(count),K_new,d_min(2),d_max(2));
[Welfare_output_5(:,count), UE_output_5(:,count),~, ~, ~] = Dynamic_Case_GSP_MEC(T,Nrun,I,R_2,J(count),K_new,d_min(3),d_max(3));
count
end

delta_RBB_1 = UE_output_3(11,:);    delta_VCG_1 = UE_output_3(15,:);
delta_RBB_2 = UE_output_4(11,:);    delta_VCG_2 = UE_output_4(15,:);
delta_RBB_3 = UE_output_5(11,:);    delta_VCG_3 = UE_output_5(15,:);

cost_RBB_1 = UE_output_3(16,:);    cost_VCG_1 = UE_output_3(20,:);
cost_RBB_2 = UE_output_4(16,:);    cost_VCG_2 = UE_output_4(20,:);
cost_RBB_3 = UE_output_5(16,:);    cost_VCG_3 = UE_output_5(20,:);
tau_max = 200e-3*ones(1,J_count);

%% 
figure(2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(1,2,1);
%-----------------------------------------------
plot(J,cost_RBB_1,'b-', 'LineWidth', 1.5); hold on; grid on;
plot(J,cost_VCG_1,'b--', 'LineWidth', 2); hold on;
plot(J,cost_RBB_2,'r-', 'LineWidth', 1.5); hold on;
plot(J,cost_VCG_2,'r--', 'LineWidth', 2); hold on;
plot(J,cost_RBB_3,'m-', 'LineWidth', 1.5); hold on;
plot(J,cost_VCG_3,'m--', 'LineWidth', 2); hold on;
legend('GSP, d_{avg}=[5,20]','VCG, d_{avg}=[5,20]','GSP, d_{avg}=[10,40]','VCG, d_{avg}=[10,40]','GSP, d_{avg}=[20,100]','VCG, d_{avg}=[20,100]');
xlabel('No. of UEs, J');
ylabel('Avg. offloading cost of UEs');
xlim([min(J),max(J)]);
%-----------------------------------------------
subplot(1,2,2);
%-----------------------------------------------
plot(J,delta_RBB_1,'b-', 'LineWidth', 1.5); hold on; grid on;
plot(J,delta_VCG_1,'b--', 'LineWidth', 2); hold on;
plot(J,delta_RBB_2,'r-', 'LineWidth', 1.5); hold on;
plot(J,delta_VCG_2,'r--', 'LineWidth', 2); hold on;
plot(J,tau_max,'g-', 'LineWidth', 2); hold on;
plot(J,delta_RBB_3,'m-', 'LineWidth', 1.5); hold on;
plot(J,delta_VCG_3,'m--', 'LineWidth', 2); hold on;
legend('GSP, d_{avg}=[5,20]','VCG, d_{avg}=[5,20]','GSP, d_{avg}=[10,40]','VCG, d_{avg}=[10,40]','GSP, d_{avg}=[20,100]','VCG, d_{avg}=[20,100]');
xlabel('No. of UEs, J');
ylabel('Avg. task execution latency');
xlim([min(J),max(J)]);




