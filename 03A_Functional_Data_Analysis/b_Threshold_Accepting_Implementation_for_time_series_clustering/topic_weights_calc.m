% Echtzeit definieren
zeit = 1996+4/12:1/12:2021+10/12;

zrmatrix = 'original_topicweights.csv';
number_topics = 120;
zrmr = readmatrix(zrmatrix,'FileType','text','NumHeaderLines',1,'Delimiter',',');
zrmr = zrmr(:,2:number_topics+1);

% weights
topic_weights_total = mean(zrmr);


zrmatrix = 'monthly_grouped_hp_filtered_trend_component_z_std_topicweights.csv';
zrhp = readmatrix(zrmatrix,'FileType','text','NumHeaderLines',1,'Delimiter',',');
zrhp = zrhp(:,2:number_topics+1);

figure
subplot(1,4,1);
plot(zeit,zrhp(:,6),'-b');
hold on
subplot(1,4,2);
plot(zeit,zrhp(:,11),'-b');
subplot(1,4,3);
plot(zeit,zrhp(:,13),'-b');
subplot(1,4,4);
plot(zeit,zrhp(:,15),'-b');

inno_labels = readmatrix('lda-labelling2_22_03_29.csv','FileType','text','NumHeaderLines',1,'Delimiter',',');
inno_grade = inno_labels(:,10)>0.5;

inno_labels([6 11 13 15],10)
inno_grade([6 11 13 15])

load('results\Dbest_120_10_22-02-12_23-49.mat','results_Dbest');  
iter = 10;

% nur 12 Cluster (2. Zeile)
cluster9 = find(results_Dbest(2,:)==9)

figure
plot(zeit,zrhp(:,cluster9(2)),'-b','Linewidth',2);
hold on
plot(zeit,zrhp(:,cluster9(14)),'-r','Linewidth',2);


load('dtw_series_time.mat','dtw_series_time');
load('dtw_series_val.mat','dtw_series_val');
load('dtm.mat','dtm');
load('zrmr.mat','zrmr');

figure

xv = dtw_series_time(cluster9(2),cluster9(14),:);
x = xv(:);
yv = dtw_series_val(cluster9(2),cluster9(14),:);
y = yv(:);
plot(zeit(x),zrmr(y,cluster9(14)),'-r','Linewidth',2);
hold on
plot(zeit,zrmr(:,cluster9(2)),':b','Linewidth',2);