fname = {'$\mathcal{N}_1^2$','$\mathcal{N}_1^3$','$\mathcal{N}_1^4$','$\mathcal{N}_1^5$','$\mathcal{N}_1^6$','$\mathcal{N}_1^7$','$\mathcal{N}_1^8$','$\mathcal{N}_1^9$','$\mathcal{N}_1^{10}$','$\mathcal{N}_1^{11}$','$\mathcal{N}_1^{12}$','$\mathcal{N}_1^{13}$'}; 
y = [100-98.94 98.94; 100-99.85  99.85; 100-99.85  99.85; 100-99.9 99.9 ;100-99.91 99.91;100-99.93 99.93;100-99.92 99.92; 100-99.94 99.94; 100-99.945 99.945;100-99.95 99.95;100-99.955 99.955;100-99.957 99.957];
%bar(y,0.5,'stacked')
bar(y(:,1),0.5,'r')
set(gca, 'XTick', 1:length(fname),'XTickLabel',fname);




legend('Active parameters')
xlabel('Hidden layer in network $N_1$') 
ylabel('% of active parameters')
