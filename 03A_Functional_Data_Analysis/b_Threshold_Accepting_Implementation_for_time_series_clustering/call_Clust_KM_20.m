% Festlegen der Parameter
% Anzahl Cluster m
% Anzahl Zeitreihen n > m
% Anzahl Restarts runs
% Next neighbours pro Schritt: nn
% Distance to neighbours: un
% Anzahl Suchschritte in TA: iter
tic;

m = [5 6 7 8 9 10 11 12 13 14 15 17 19 21 23 25 27 29 31 33 35]';
%m = [7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53]';
%m= 9;

n = 120*ones(size(m,1),1);
runs = 30*ones(size(m,1),1);
nn = 2*ones(size(m,1),1);
un = 2*ones(size(m,1),1);
iter = 100*ones(size(m,1),1);
%iter = 100;

parameters=[m n runs nn un iter];

% parameters = [  6 120 10 2 2 2000000
%                 7 120 10 2 2 2000000
%                 8 120 10 2 2 2000000
%                 9 120 10 2 2 2000000
%                10 120 10 2 2 2000000
%                11 120 10 2 2 2000000
%                12 120 10 2 2 2000000
%                13 120 10 2 2 2000000
%                14 120 10 2 2 2000000
%                15 120 10 2 2 2000000
%                16 120 10 2 2 2000000
%                17 120 10 2 2 2000000
%                18 120 10 2 2 2000000
%                19 120 10 2 2 2000000
%                20 120 10 2 2 2000000
%                21 120 10 2 2 2000000
%                22 120 10 2 2 2000000
%                23 120 10 2 2 2000000
%                24 120 10 2 2 2000000];
% 
% n = 120;

results = [parameters zeros(size(parameters,1),2)];
results_best = zeros(size(parameters,1),1);
results_best1 = zeros(size(parameters,1),1);
results_objf_mean = zeros(size(parameters,1),1);
results_objf_var = zeros(size(parameters,1),1);
results_Dbest = zeros(size(parameters,1),parameters(1,2));

% Geglätte Zeitreihen einlesen
zrmatrix = 'monthly_grouped_hp_filtered_trend_component_z_std_topicweights.csv';
zeit = 1996+4/12:1/12:2021+10/12;
%[zrmr, dtm, dtm_series_time, dtm_series_val] = DTW_Berechnung(zrmatrix,n);

% load
load('dtw_series_time.mat','dtw_series_time');
load('dtw_series_val.mat','dtw_series_val');
load('dtm.mat','dtm');
load('zrmr.mat','zrmr');

%profile on;

[zeilen,spalten] = size(parameters);
for i=1:zeilen
    m = parameters(i,1);
    n = parameters(i,2);
    if n <= m
        fprintf('Attention: number of cluster larger than number of series')
    end
    runs = parameters(i,3); 
    nn = parameters(i,4);
    un = parameters(i,5);
    iter = parameters(i,6);
    %fprintf('\n %3i %3i %3i %2i %10i',m,n,runs,nn,un,iter);
    
    % [best,best1,objf_mean,objf_var,Dbest]=Clust_TA_1(m,n,runs,nn,un,iter,dtm);
    [best,best1,objf_mean,objf_var,Dbest]=Clust_KM_1(m,n,runs,nn,un,iter,zrmr');

    results_best(i) = best;
    results_best1(i) = best1; 
    results_objf_mean(i) = objf_mean;
    results_objf_var(i) = objf_var;
    results_Dbest(i,:) = Dbest(:,1)';

    %fprintf(f18,'\n %3i %3i %2i %10i %8.5f %8.5f',m,n,nn,iter,best,best1);
    %fprintf('\n %3i %3i %2i %10i %8.5f %8.5f',m,n,nn,iter,best,best1);
    %fprintf(f18,'\n --- best clustering ---\n');
    %fprintf('\n --- best clustering --- \n');
     [Dr,Dc] = size(Dbest);
     for j = 1:Dr
         %fprintf(f18,'%3i',Dbest(j,:));
         %fprintf(f18,'\n');
         %fprintf('%3i',Dbest(j,:));
         %fprintf('\n');
     end
     % filemat = ['results\Dbest_',mat2str(m),'_',mat2str(n),'_',mat2str(iter/1000000),'_',datestr(datetime,formatOut),'.mat']
     % save(filemat,'Dbest');
end

results(:,7:10) = [results_best, results_best1, results_objf_mean, results_objf_var];

formatOut='yy-mm-dd_HH-MM';
filer = ['results\Clust_KM_2.0_',datestr(datetime,formatOut),'.txt']
f18 = fopen(filer,'w');
fprintf(f18,'\n--- Results for TA Clustering -------------------------------------');
%fprintf(f18,'\n iter = %8i',iter);
fprintf(f18,'\n');

for iil=1:size(results,1)
    fprintf(f18,'\n %3i %3i %3i %2i %2i %10i %12.4f %12.4f %12.4f %12.4f',results(iil,:));
end

timem= toc;
fprintf(f18,'\n\n*** total CPU-seconds: %12.4f',timem);
fprintf('\n total CPU-seconds: %12.4f',timem);
fprintf('\n');

fprintf(f18,'\n');
f18suc = fclose(f18); 

     
filemat = ['results\Dbest_KM_',mat2str(n(1)),'_',mat2str(iter(1)/1000000),'_',datestr(datetime,formatOut),'.mat'];
save(filemat,'results_Dbest');

figure
hold on
plot(results(:,1),results(:,9),':b');
plot(results(:,1),results(:,7),'-g');
conf_lb = results(:,9) - 2*(results(:,10));
conf_ub = results(:,9) + 2*(results(:,10));
plot(results(:,1),conf_lb,'--r');
plot(results(:,1),conf_ub,'--r');
legend('mean','best','-2sigma','+2sigma');

formatOut2='yy-mm-dd:HH-MM ';
title_text = ['Ellbowplot\_KM\_',datestr(datetime,formatOut2),'\_',int2str(iter(1)),'\_',int2str(m)];
title(title_text);

figuren = ['results\Clust_KM_20_dtm_Ellbow_',datestr(datetime,formatOut),'_',int2str(iter(1)),'_',int2str(m),'.jpg'];
print(figuren,'-djpeg');
%profile viewer;
%profile off;

% DTW_Grafiken_Test
