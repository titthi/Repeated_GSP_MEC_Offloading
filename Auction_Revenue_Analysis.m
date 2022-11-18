%%
clc; clear all; close all;
tic % measuring start time



%% Run Main function and plot graphs
T = 50;
Nrun=100;
J = 150;             
R = [80; 80];         % case-1
%--------------------------------------------------------------------------
count = 0;
[Welfare_output, ~, MEC_output, Bid_output, Price_output] = Static_Case_GSP_MEC(T,Nrun,R,J);
count = count + 1



toc 


%----------------------------------------------------------
%% Find convergence point for RBB bids and prices
%----------------------------------------------------------
b_RBB = Bid_output(3:4,:);      b_BB = Bid_output(5:6,:);       b_AB = Bid_output(7:8,:);       b_CB = Bid_output(9:10,:);       b_VCG = Bid_output(11:12,:);
p_RBB = Price_output(1,:);      p_BB = Price_output(2,:);       p_AB = Price_output(3,:);       p_CB = Price_output(4,:);       p_VCG = Price_output(5,:);

u_RBB = MEC_output(1:2,:);      u_CB = MEC_output(7:8,:);       u_VCG = MEC_output(9:10,:);
z_RBB = Welfare_output(6,:);    z_CB = Welfare_output(9,:);     z_VCG = Welfare_output(10,:);
pm_RBB = Welfare_output(11,:);   pm_BB = Welfare_output(12,:);    pm_AB = Welfare_output(13,:);    pm_CB = Welfare_output(14,:);    pm_VCG = Welfare_output(15,:);


bid_diff = zeros(1,T);  bid_conv = zeros(2,1);
price_diff = zeros(1,T);    price_conv = zeros(2,1);
%-----------------------------------------------
for i = 1:2
    for t = 1:T-1
        bid_diff(t) = b_RBB(i,t) - b_RBB(i,t+1);
        price_diff(t) = abs(p_RBB(t) - p_RBB(t+1));
        if(bid_diff(t)<= 1e-5)
            bid_conv(i) = t+1;
            break;
        end
        if(price_diff(t) <= 0.0001)
            price_conv(i) = t+1;
            break;
        end
    end
end

%-----------------------------------------------

%------------------------------------------------
b_RBB_mean = mean(b_RBB);      b_BB_mean = mean(b_BB);       b_AB_mean = mean(b_AB);       b_CB_mean = mean(b_CB);       b_VCG_mean = mean(b_VCG);

[b_max_x(1),b_max_y(1)] = max(b_RBB_mean);
[b_max_x(2),b_max_y(2)] = max(b_BB_mean);
[b_max_x(3),b_max_y(3)] = max(b_CB_mean);
[b_max_x(4),b_max_y(4)] = max(b_AB_mean);
[b_max_x(5),b_max_y(5)] = max(b_VCG_mean);

[p_max_x(1),p_max_y(1)] = max(p_RBB);
[p_max_x(2),p_max_y(2)] = max(p_BB);
[p_max_x(3),p_max_y(3)] = max(p_CB);
[p_max_x(4),p_max_y(4)] = max(p_AB);
[p_max_x(5),p_max_y(5)] = max(p_VCG);

[pm_max_x(1),pm_max_y(1)] = max(pm_RBB);
[pm_max_x(2),pm_max_y(2)] = max(pm_BB);
[pm_max_x(3),pm_max_y(3)] = max(pm_CB);
[pm_max_x(4),pm_max_y(4)] = max(pm_AB);
[pm_max_x(5),pm_max_y(5)] = max(pm_VCG);
%-------------------------------------------

%% 
figure(1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-----------------------------------------------
subplot(1,2,1);
%-----------------------------------------------
plot(1:T,b_RBB(1,:),'b-', 'LineWidth', 1.5); hold on; grid on;
plot(1:T,b_RBB(2,:),'b--', 'LineWidth', 1.5); hold on;
plot(1:T,b_CB(1,:),'r-', 'LineWidth', 1.5); hold on;
plot(1:T,b_CB(2,:),'r--', 'LineWidth', 1.5); hold on;
plot(1:T,b_VCG(1,:),'m-', 'LineWidth', 1.5); hold on;
plot(1:T,b_VCG(2,:),'m--', 'LineWidth', 1.5); hold on;
legend('b_1 (RBB)','b_2 (RBB)','b_1 (CB)','b_2 (CB)','b_1 (VCG)','b_2 (VCG)');
xlabel('Auction rounds, t');
ylabel('Bids ($/VM-hr)');
xlim([1,T]);
%-----------------------------------------------
subplot(1,2,2);
%-----------------------------------------------
plot(1:T,p_RBB,'b-', 'LineWidth', 1.5); hold on; grid on;
plot(1:T,p_CB,'r-', 'LineWidth', 1.5); hold on;
plot(1:T,p_VCG,'m-', 'LineWidth', 1.5); hold on;
legend('RBB','CB','VCG');
xlabel('Auction rounds, t');
ylabel('Allocation price, ($/VM-hr)');
xlim([1,T]);


%% 
figure(2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-----------------------------------------------
subplot(1,2,1);
%-----------------------------------------------
plot(1:T,u_RBB(1,:),'b-', 'LineWidth', 1.5); hold on; grid on;
plot(1:T,u_RBB(2,:),'b--', 'LineWidth', 1.5); hold on;
plot(1:T,u_CB(1,:),'r-', 'LineWidth', 1.5); hold on;
plot(1:T,u_CB(2,:),'r--', 'LineWidth', 1.5); hold on;
plot(1:T,u_VCG(1,:),'m-', 'LineWidth', 1.5); hold on;
plot(1:T,u_VCG(2,:),'m--', 'LineWidth', 1.5); hold on;
legend('u_1 (RBB)','u_2 (RBB)','u_1 (CB)','u_2 (CB)','u_1 (VCG)','u_2 (VCG)');
xlabel('Auction rounds, t');
ylabel('Profit of servers ($/VM-hr)');
xlim([1,T]);
%-----------------------------------------------
subplot(1,2,2);
%-----------------------------------------------
plot(1:T,z_RBB,'b-', 'LineWidth', 1.5); hold on; grid on;
plot(1:T,z_CB,'r-', 'LineWidth', 1.5); hold on;
plot(1:T,z_VCG,'m-', 'LineWidth', 1.5); hold on;
legend('RBB','CB','VCG');
xlabel('Auction rounds, t');
ylabel('Total allocation valuation ($/VM-hr)');
xlim([1,T]);


%-----------------------------------------------
figure(3)
%-----------------------------------------------
subplot(1,2,1);
%-----------------------------------------------
bid_price = [b_max_x(1) b_max_x(2) b_max_x(3) b_max_x(4) b_max_x(5);  p_max_x(1) p_max_x(2) p_max_x(3) p_max_x(4) p_max_x(5)];
bar(bid_price); grid on;
legend('RBB','BB','CB','AB','VCG')
ylabel('($/VM-hour)');
%-----------------------------------------------
subplot(1,2,2);
%-----------------------------------------------
pm = [pm_max_x(1); pm_max_x(2); pm_max_x(3); pm_max_x(4); pm_max_x(5)];
bar(1,pm_max_x(1)); hold on; grid on;   
bar(2,pm_max_x(2)); hold on;
bar(3,pm_max_x(3)); hold on;    
bar(4,pm_max_x(4)); hold on;
bar(5,pm_max_x(5)); hold on;       
ylabel('profit margin ratio(%)');
%-----------------------------------------------

