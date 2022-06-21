x=[2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26];
y=[0.065;0.039;0.033;0.0314;0.0309;0;0;0;0.00955;0.007938;0.003446;0.00301;0.00270;0.0025;0.002;0.0016;0.00148;0.00139;0.00139;0.00126;0.00113;0.00109;0.001046;0.001046;0.001046];

bar(x,y,0.4)
ylabel('Test loss (Relative error)')
title('Summary of architecture adaptation results')
ylim([0;0.04])
yticks([0.01,0.02,0.03,0.04,0.05,0.06,0.07])
xticks([2;3;4;5;6])


set(0,'defaulttextinterpreter','latex');  
set(0, 'defaultAxesTickLabelInterpreter','latex');  
set(0, 'defaultLegendInterpreter','latex');



fname = {'$\mathcal{N}_1^2$','$\mathcal{N}_1^3$','$\mathcal{N}_1^4$','$\mathcal{N}_1^5$','$\mathcal{N}_1^6$','$\mathcal{N}_1^7$','$\mathcal{N}_1^8$','$\mathcal{N}_1^9$','$\mathcal{N}_1^{10}$','$\mathcal{N}_1^{11}$', '$\mathcal{N}_1^{12}$','$\mathcal{N}_1^{13}$','$\mathcal{N}_1^{14}$'}; 
x = [1:13]; y = [74.8;77.9;86.8;87;87.6;87.7;88.4;89.1;89.3;89.5;89.5;89.5;89.5]; 
bar(x,y)
set(gca, 'XTick', 1:length(fname),'XTickLabel',fname);

% NEW PLOTTING FUNCTION

fname = {'$\mathcal{N}_1^2$','','','$\mathcal{N}_1^3$','','','$\mathcal{N}_1^4$','','','$\mathcal{N}_1^5$','','','$\mathcal{N}_1^6$','','','$\mathcal{N}_1^7$','','','$\mathcal{N}_1^8$','','','$\mathcal{N}_1^9$','','','$\mathcal{N}_1^{10}$','','','$\mathcal{N}_1^{11}$','','', '$\mathcal{N}_1^{12}$','','','$\mathcal{N}_1^{13}$','','','$\mathcal{N}_1^{14}$','','','','','','','','','','','$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$','','','$\mathcal{N}_6$','','','$\mathcal{N}_7$','','','$\mathcal{N}_8$','','','$\mathcal{N}_9$','','','$\mathcal{N}_{10}$','','','$\mathcal{N}_{11}$','','','$\mathcal{N}_{12}$','','','$\mathcal{N}_{13}$','','','$\mathcal{N}_{14}$'};
x = [1:3:45]; y = (1/100)*[74.8;77.9;86.8;87;87.6;87.7;88.4;89.1;89.3;89.5;89.5;89.5;89.5;0;0]; 
bar(x,y,0.4,'y')


hold on

%fname1 = {'$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$','','','$\mathcal{N}_6$','','','$\mathcal{N}_7$','','','$\mathcal{N}_8$','','','$\mathcal{N}_9$','','','$\mathcal{N}_{10}$','','','$\mathcal{N}_{11}$','','','$\mathcal{N}_{12}$','','','$\mathcal{N}_{13}$','','','$\mathcal{N}_{14}$'}; 
x = [48:3:84]; y = (1/100)*[94.6;95.5;95.9;96.3;96.4;96.6;96.7;96.7;96.7;96.8;96.8;96.9;96.9]; 
bar(x,y,0.4,'r')
set(gca, 'XTick',1:length(fname),'XTickLabel',fname);
set(gcf, 'Position', [0,0,900,400]);
h=gca; h.XAxis.TickLength = [0 0];
ylim([0.7,1.1])
yticks([0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1])
hold on 

plot([0:3:88],0.96*ones(length([0:3:88]),1),'--','Color','b');

hold on

plot([0:3:88],0.959*ones(length([0:3:88]),1),'-.','Color','b');

hold on

plot([0:3:88],0.959*ones(length([0:3:88]),1),'-*','Color','b');

hold on

legend({'Accuracy by Algorithm I (89.5 %)','Accuracy by Algorithm 2 (96.9 %)','Baseline network (96 %)', 'Baseline+ Algo II  (95.9 %)', 'Forward Thinking-Hettinger et al. (95.9 %)'},'NumColumns',2)
xlabel('Neural Network $\mathcal{N}$')
ylabel('Test Accuracy')