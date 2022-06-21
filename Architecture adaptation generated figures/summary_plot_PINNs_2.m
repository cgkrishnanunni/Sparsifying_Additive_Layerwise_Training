set(0,'defaulttextinterpreter','latex');  
set(0, 'defaultAxesTickLabelInterpreter','latex');  
set(0, 'defaultLegendInterpreter','latex');



% NEW PLOTTING FUNCTION

fname = {'$\mathcal{N}_1^2$','','','$\mathcal{N}_1^3$','','','$\mathcal{N}_1^4$','','','$\mathcal{N}_1^5$','','','$\mathcal{N}_1^6$','','','','','','','','','','','$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$','','','$\mathcal{N}_6$','','','$\mathcal{N}_7$'};

x = [1:3:21]; y = [0.359;0.0475;0.00113;0.000513;0.000327;0;0]; 

bar(x,y,0.4,'y')


hold on

%fname1 = {'$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$','','','$\mathcal{N}_6$','','','$\mathcal{N}_7$','','','$\mathcal{N}_8$','','','$\mathcal{N}_9$','','','$\mathcal{N}_{10}$','','','$\mathcal{N}_{11}$','','','$\mathcal{N}_{12}$','','','$\mathcal{N}_{13}$','','','$\mathcal{N}_{14}$'}; 
x = [24:3:39]; y = [0.00015;0.000083;0.000074;0.000069;0.000054;0.000051]; 
bar(x,y,0.4,'r')
set(gca, 'XTick',1:length(fname),'XTickLabel',fname);
set(gcf, 'Position', [0,0,900,400]);
h=gca; h.XAxis.TickLength = [0 0];



ylim([0.0,0.001])
%yticks([5 10 12.5 15 20 25 30 35 40 ])
hold on 

plot([0:3:44],0.0001*ones(length([0:3:44])),'--','Color','b','LineWidth',0.1);

legend('Relative $L^2$ error by Algorithm I','Relative $L^2$ error by Algorithm 2','Relative $L^2$ error by Baseline network')
xlabel('Neural Network $\mathcal{N}$')
ylabel('Relative $L^2$ error')
hold on

x = [0.181 , 0.181];
y = [0.88 0.92];
annotation('arrow',x,y)


hold on
x = [0.236 , 0.236];
y = [0.88 0.92];
annotation('arrow',x,y)


hold on
x = [0.285 , 0.285];
y = [0.88 0.92];
annotation('arrow',x,y)

set(gca,'FontSize',18)