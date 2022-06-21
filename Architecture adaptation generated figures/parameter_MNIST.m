fname = {'$\mathcal{N}_1^2$','$\mathcal{N}_1^3$','$\mathcal{N}_1^4$','$\mathcal{N}_1^5$','$\mathcal{N}_1^6$','$\mathcal{N}_1^7$','$\mathcal{N}_1^8$','$\mathcal{N}_1^9$','$\mathcal{N}_1^{10}$','$\mathcal{N}_1^{11}$','$\mathcal{N}_1^{12}$','$\mathcal{N}_1^{13}$','$\mathcal{N}_1^{14}$'}; 
y = [75.7 24.3; 35 65; 42.4 57.6; 31.2 68.8; 32.2 67.8; 29.6 70.4; 30.8 69.2; 28.4 71.6; 25.8 74.2; 22 78; 18.8 81.2; 18.6 81.4; 17.7 82.3 ];
bar(y,0.5,'stacked')
set(gca, 'XTick', 1:length(fname),'XTickLabel',fname);




legend('Active parameters','Inactive parameters')
xlabel('Hidden layer in network $N_1$') 
ylabel('% of active/inactive params.')
ylim([0, 140])
set(gca,'FontSize',15)