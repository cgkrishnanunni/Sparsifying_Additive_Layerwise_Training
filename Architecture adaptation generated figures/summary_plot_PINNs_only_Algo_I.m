set(0,'defaulttextinterpreter','latex');  
set(0, 'defaultAxesTickLabelInterpreter','latex');  
set(0, 'defaultLegendInterpreter','latex');



% NEW PLOTTING FUNCTION

fname = {'$\mathcal{N}_1^2$','','','$\mathcal{N}_1^3$','','','$\mathcal{N}_1^4$','','','$\mathcal{N}_1^5$','','','$\mathcal{N}_1^6$','','','$\mathcal{N}_1^7$', '','', '$\mathcal{N}_1^8$'};

x = [1:3:21]; y = [0.1723;0.0254;0.004974;0.0006649;0.000266;0.00006722;0.0000662]; 

bar(x,y,0.4,'y')


hold on

%fname1 = {'$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$','','','$\mathcal{N}_6$','','','$\mathcal{N}_7$','','','$\mathcal{N}_8$','','','$\mathcal{N}_9$','','','$\mathcal{N}_{10}$','','','$\mathcal{N}_{11}$','','','$\mathcal{N}_{12}$','','','$\mathcal{N}_{13}$','','','$\mathcal{N}_{14}$'}; 
%x = [24:3:45]; y = [0.0001897;0.0001573;0.0001381;0.00006328;0.00005879;0.00005402;0.00005165;0.00002669]; 
%bar(x,y,0.4,'r')
set(gca, 'XTick',1:length(fname),'XTickLabel',fname);
set(gcf, 'Position', [0,0,900,400]);
h=gca; h.XAxis.TickLength = [0 0];



ylim([0.0,0.001])
%yticks([5 10 12.5 15 20 25 30 35 40 ])
hold on 

plot([0:3:50],0.00012*ones(length([0:3:50]),1),'--','Color','b');

hold on

plot([0:3:50],0.0921*ones(length([0:3:50]),1),'-*','Color','b');


hold on

legend('Relative $L^2$ error by Algorithm I','Relative $L^2$ error by Baseline network', 'Forward Thinking ($9.21\times 10^{-3}$)')
xlabel('Neural Network $\mathcal{N}$')
ylabel('Relative $L^2$ error')
hold on

x = [0.178 , 0.178];
y = [0.88 0.92];
annotation('arrow',x,y)


hold on
x = [0.222 , 0.222];
y = [0.88 0.92];
annotation('arrow',x,y)


hold on
x = [0.27 , 0.27];
y = [0.88 0.92];
annotation('arrow',x,y)

set(gca,'FontSize',18)