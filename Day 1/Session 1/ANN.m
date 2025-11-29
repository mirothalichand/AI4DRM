clear all
close all
load final_data.mat
target = log10(Sa);
% input
input = [Mw Repi log10(Rhyp) log10(Vs30) FM];
x = input';
t = target';
% training function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hidden =size(input,2);
net = fitnet(hidden,trainFcn);
net.layers{1}.transferFcn = 'tansig';%logsig
net.layers{2}.transferFcn = 'purelin';

% Choose Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio  = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

% View the Network
view(net)
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (true)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'ann_gmpe_3','MatrixOnly','yes');
    y = ann_gmpe_3(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end
% y = Check_ANN_BB4(x);
err=y-t;
SD=std(err,0,2);
MSR=(sum(err.^2,2)/length(y));
PP=1-(MSR./std(t,0,2).^2);
r1=sum((y-mean(y,2)).^2,2);
r2=sum((t-mean(t,2)).^2,2);
R=sum(((y-mean(y,2)).*(t-mean(t,2))),2)./(sqrt(r1.*r2));
% for total data
y_t=y(:);
t_t=t(:);
err_t=y_t-t_t;
SD_t=std(err_t);
MSR_t=(sum(err_t.^2)/length(y_t));
PP_t=1-(MSR_t/std(t_t).^2);
r1_t=sum((y_t-mean(y_t)).^2);
r2_t=sum((t_t-mean(t_t)).^2);
R_t=sum((y_t-mean(y_t)).*(t_t-mean(t_t)))/(sqrt(r1_t.*r2_t));
perfor=[R PP MSR SD]';
perfor_t=[R_t PP_t MSR_t SD_t];

%% Parametric Plots
%% plot of ANN for diffrent parametric study

figure(1)
hold on;
% load period
M=[6.0 7.0];
R = [10 22.36 50 111.80 250];
Vs30=760; %% SC=0 for rock ground, SC=1 for firm ground, SC=2 for soft soil
F = [1 3];
Flag = 2;
co=(colormap(hsv(length(R))));
set(groot,'defaultAxesColorOrder',co)
load nga_sub.mat T
T = T;
D = 10;

for k=1:1:length(F)
for j=1:1:length(M)
for i=1:1:length(R)
    kk = [M(j) R(i) log10(R(i)) log10(Vs30) D F(k) Flag];
    A1=ann_gmpe_NE3(kk');
    PGA1(k,j,i,:)=10.^(A1);
end
end
end

PGA11(:,:)=PGA1(1,1,:,:);
subplot(2,2,1)
loglog(T,PGA11,'LineWidth',1.5);
xlabel('Period (s)','FontWeight','bold','FontSize',12)
ylabel('Spectral Acceleration(g)','FontWeight','bold','FontSize',12)
title('Regional: M_w=6.0 F=Strike Slip, SC=1','FontWeight','bold','FontSize',12)
set(gca,'FontWeight','bold','FontSize',10)
xlim([0 4])
box on; grid on;


PGA12(:,:)=PGA1(1,2,:,:);
subplot(2,2,2)
loglog(T,PGA12,'LineWidth',1.5);
xlabel('Period (s)','FontWeight','bold','FontSize',12)
ylabel('Spectral Acceleration(g)','FontWeight','bold','FontSize',12)
set(gca,'FontWeight','bold','FontSize',10)
xlim([0 4])
box on; grid on;

title('M_w=7.0 F=Strike Slip,  SC=1','FontWeight','bold','FontSize',12)

PGA21(:,:)=PGA1(2,1,:,:);
subplot(2,2,3)
loglog(T,PGA21,'LineWidth',1.5);
xlabel('Period (s)','FontWeight','bold','FontSize',12)
ylabel('Spectral Acceleration(g)','FontWeight','bold','FontSize',12)
set(gca,'FontWeight','bold','FontSize',10)
box on; grid on;
xlim([0 4])
title('M_w=6.0 F=Reverse, SC=1','FontWeight','bold','FontSize',12)


PGA22(:,:)=PGA1(2,2,:,:);
subplot(2,2,4)
loglog(T,PGA22,'LineWidth',1.5);
xlabel('Period (s)','FontWeight','bold','FontSize',12)
ylabel('Spectral Acceleration(g)','FontWeight','bold','FontSize',12)
set(gca,'FontWeight','bold','FontSize',10)
box on; grid on;
xlim([0 4])
title('M_w=7.0 F=Reverse,  SC=1','FontWeight','bold','FontSize',12)
legend R_{hyp}10km R_{hyp}22.36km R_{hyp}50km R_{hyp}111.80km  R_{hyp}250km
clear PGA11 PGA12 PGA21 PGA22

%%

figure(3)
hold on;
% load period
M=[6.0 7.0];
R = [10 22.36 50 111.80 250];
Vs30=760; %% SC=0 for rock ground, SC=1 for firm ground, SC=2 for soft soil
F = [1 3];
Flag = 2;
co=(colormap(hsv(length(R))));
set(groot,'defaultAxesColorOrder',co)
load nga_sub.mat T
T = T;
D = 70;

for k=1:1:length(F)
for j=1:1:length(M)
for i=1:1:length(R)
    kk = [M(j) R(i) log10(R(i)) log10(Vs30) D F(k) Flag];
    A1=ann_gmpe_NE3(kk');
    PGA1(k,j,i,:)=10.^(A1);
end
end
end

PGA11(:,:)=PGA1(1,1,:,:);
subplot(2,2,1)
loglog(T,PGA11,'LineWidth',1.5);
xlabel('Period (s)','FontWeight','bold','FontSize',12)
ylabel('Spectral Acceleration(g)','FontWeight','bold','FontSize',12)
title(' M_w=6.0 F=Strike Slip V_{s30}=760m/s, ','FontWeight','bold','FontSize',12)
set(gca,'FontWeight','bold','FontSize',10)
xlim([0 4])
box on; grid on;


PGA12(:,:)=PGA1(1,2,:,:);
subplot(2,2,2)
loglog(T,PGA12,'LineWidth',1.5);
xlabel('Period (s)','FontWeight','bold','FontSize',12)
ylabel('Spectral Acceleration(g)','FontWeight','bold','FontSize',12)
set(gca,'FontWeight','bold','FontSize',10)
xlim([0 4])
box on; grid on;

title('M_w=7.0 F=Strike Slip V_{s30}=760m/s','FontWeight','bold','FontSize',12)

PGA21(:,:)=PGA1(2,1,:,:);
subplot(2,2,3)
loglog(T,PGA21,'LineWidth',1.5);
xlabel('Period (s)','FontWeight','bold','FontSize',12)
ylabel('Spectral Acceleration(g)','FontWeight','bold','FontSize',12)
set(gca,'FontWeight','bold','FontSize',10)
box on; grid on;
xlim([0 4])
title('M_w=6.0 F=Reverse V_{s30}=760m/s','FontWeight','bold','FontSize',12)


PGA22(:,:)=PGA1(2,2,:,:);
subplot(2,2,4)
loglog(T,PGA22,'LineWidth',1.5);
xlabel('Period (s)','FontWeight','bold','FontSize',12)
ylabel('Spectral Acceleration(g)','FontWeight','bold','FontSize',12)
set(gca,'FontWeight','bold','FontSize',10)
box on; grid on;
xlim([0 4])
title('M_w=7.0 F=Reverse V_{s30}=760m/s','FontWeight','bold','FontSize',12)
legend R_{epi}10km R_{epi}22.36km R_{epi}50km R_{epi}111.80km  R_{epi}250km
clear PGA11 PGA12 PGA21 PGA22
