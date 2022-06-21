


set(0,'defaulttextinterpreter','latex');  
set(0, 'defaultAxesTickLabelInterpreter','latex');  
set(0, 'defaultLegendInterpreter','latex');



% NEW PLOTTING FUNCTION

fname = {'$\mathcal{N}_1^2$','','','$\mathcal{N}_1^3$','','','$\mathcal{N}_1^4$','','','$\mathcal{N}_1^5$','','','$\mathcal{N}_1^6$','','','$\mathcal{N}_1^7$','','','$\mathcal{N}_1^8$','','','$\mathcal{N}_1^9$','','','','','','','','','','','$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$'};

x = [1:3:30]; y = [341.4;109.3;27.7;25.27;24.52;24.3;24.21;24.2;0;0]; 

bar(x,y,0.4,'y')


hold on

%fname1 = {'$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$','','','$\mathcal{N}_6$','','','$\mathcal{N}_7$','','','$\mathcal{N}_8$','','','$\mathcal{N}_9$','','','$\mathcal{N}_{10}$','','','$\mathcal{N}_{11}$','','','$\mathcal{N}_{12}$','','','$\mathcal{N}_{13}$','','','$\mathcal{N}_{14}$'}; 
x = [33:3:42]; y = [9.74;7.85;7.50;7.11]; 
bar(x,y,0.4,'r')
set(gca, 'XTick',1:length(fname),'XTickLabel',fname);
set(gcf, 'Position', [0,0,900,400]);
h=gca; h.XAxis.TickLength = [0 0];
ylim([0.0,45])
yticks([5 11.1 15 20 25 28 30 35 40 ])
hold on 


hold on
plot(0:3:47,11.1*ones(length(0:3:47),1),'--','Color','b');
hold on

plot(0:3:47,9.7*ones(length(0:3:47),1),'-o');

hold on

plot(0:3:47,28*ones(length(0:3:47),1),'-*');






legend({'Test loss by Algo I','Test loss by Algo 2','Baseline network', 'Baseline Network+ Algo II', 'Forward Thinking [18]'},'NumColumns',2)
xlabel('Neural Network $\mathcal{N}$')
ylabel('Test loss')
hold on

x = [0.177 , 0.177];
y = [0.88 0.92];
annotation('arrow',x,y)
hold on
x = [0.230 , 0.230];
y = [0.88 0.92];
annotation('arrow',x,y)
set(gca,'FontSize',17)