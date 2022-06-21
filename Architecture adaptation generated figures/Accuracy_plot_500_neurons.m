set(0,'defaulttextinterpreter','latex');  
set(0, 'defaultAxesTickLabelInterpreter','latex');  
set(0, 'defaultLegendInterpreter','latex');




% NEW PLOTTING FUNCTION

fname = {'$\mathcal{N}_1^2$','','','$\mathcal{N}_1^3$','','','$\mathcal{N}_1^4$','','','$\mathcal{N}_1^5$','','','$\mathcal{N}_1^6$','','','$\mathcal{N}_1^7$','','','$\mathcal{N}_1^8$','','','$\mathcal{N}_1^9$','','','$\mathcal{N}_1^{10}$','','','$\mathcal{N}_1^{11}$','','', '$\mathcal{N}_1^{12}$','','','$\mathcal{N}_1^{13}$','','','','','','','','','','','$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$'};
x = [1:3:42]; y = (1/100)*[64.7;68;82.8;84.0;84.5; 84.6;84.9;85.3;86;86.2; 86.5; 87;0;0]; 
bar(x,y,0.4,'y')


hold on

%fname1 = {'$\mathcal{N}_2$','','','$\mathcal{N}_3$','','','$\mathcal{N}_4$','','','$\mathcal{N}_5$','','','$\mathcal{N}_6$','','','$\mathcal{N}_7$','','','$\mathcal{N}_8$','','','$\mathcal{N}_9$','','','$\mathcal{N}_{10}$','','','$\mathcal{N}_{11}$','','','$\mathcal{N}_{12}$','','','$\mathcal{N}_{13}$','','','$\mathcal{N}_{14}$'}; 

x = [45:3:54]; y = (1/100)*[98.6; 98.6; 98.6; 98.7]; 
bar(x,y,0.4,'r')
set(gca, 'XTick',1:length(fname),'XTickLabel',fname);
set(gcf, 'Position', [0,0,900,400]);
h=gca; h.XAxis.TickLength = [0 0];
ylim([0.6,1.2])
yticks([0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1])
hold on 

plot([0:3:57],0.982*ones(length([0:3:57]),1),'--','Color','b');

hold on

plot([0:3:57],0.982*ones(length([0:3:57]),1),'-.','Color','b');

hold on

plot([0:3:57],0.983*ones(length([0:3:57]),1),'-*','Color','b');

hold on

plot([0:3:57],0.986*ones(length([0:3:57]),1),'-o','Color','b');

legend({'Accuracy by Algorithm I (87 %)','Accuracy by Algorithm 2 (98.7 %)','Baseline network (98.2 %)', 'Baseline+ Algo II  (98.2 %)', 'Forward Thinking-Hettinger et al. (98.3 %)', 'LeNet-5 (CNN) (98.6 %)'},'NumColumns',2)
xlabel('Neural Network $\mathcal{N}$')
ylabel('Test Accuracy')

