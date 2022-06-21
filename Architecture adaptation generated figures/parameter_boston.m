fname = {'$\mathcal{N}_1^2$','$\mathcal{N}_1^3$','$\mathcal{N}_1^4$','$\mathcal{N}_1^5$','$\mathcal{N}_1^6$','$\mathcal{N}_1^7$','$\mathcal{N}_1^8$','$\mathcal{N}_1^9$'}; 
y = [98.8 1.2 ; 85 15 ; 78.5 21.5 ; 57 43; 48.4 51.6; 34.5 65.5; 31.4 68.6; 25.5 74.5 ];
bar(y,0.5,'stacked')
set(gca, 'XTick', 1:length(fname),'XTickLabel',fname);




legend('Active parameters','Inactive parameters')
xlabel('Hidden layer in network $N_1$') 
ylabel('% of active/inactive params')
ylim([0, 140])
yticks([0,20,40,60,80,100,120])
set(gca,'FontSize',18)