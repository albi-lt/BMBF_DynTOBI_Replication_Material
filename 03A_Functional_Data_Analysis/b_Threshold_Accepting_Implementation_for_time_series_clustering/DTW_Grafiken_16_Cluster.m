% Echtzeit definieren
zeit = 1996+4/12:1/12:2021+10/12;

% load
load('dtw_series_time.mat','dtw_series_time');
load('dtw_series_val.mat','dtw_series_val');
load('dtm.mat','dtm');
load('zrmr.mat','zrmr');

load('results\Dbest_120_0.1_25-01-22_07-59.mat','results_Dbest');  

inno_labels = readmatrix('lda-labelling2_22_03_29.csv','FileType','text','NumHeaderLines',1,'Delimiter',',');
inno_grade = inno_labels(:,10)>0.5;


% Nur Ergebnisse für 1
for Di = 1:1
    Dbest = results_Dbest(Di,:)';
    m = max(Dbest);
    nr_rows = round(sqrt(m));
    nr_cols = ceil(m/nr_rows);

% Nur Cluster 16 dafür detaillierter
ci = 16;
    
% Abbildung "echte" Zeitreihen
figure

hold on
indices = find(Dbest == ci);
for cii = 1:length(indices)
    plot(zeit,zrmr(:,indices(cii)));
    inno_labels(indices(cii),10)
end
cmap = colororder;


% formatOut2='yy-mm-dd-HH-MM';
% figuren = ['results\FR_',datestr(datetime,formatOut2),'_LDA_real_',int2str(iter(1)),'_',int2str(m),'.pdf'];
% print(figuren,'-dpdf');
 
% Abbildung der Word Clouds für die Prototypen
 
figure
[hand,posi] = tight_subplot(nr_rows,nr_cols,[.02 -.2],[.05 .01],[.01 .01]);

ser_ind = find(Dbest == ci);
for cii = 1:length(ser_ind)
    axes(hand(cii));
    wc_name = ['2022_02_16_Wordclouds_120_Topics_LDA\allTopics_Seite_',sprintf('%03d',ser_ind(cii)),'.tiff'];
    [imx,im_cmap] = imread(wc_name);
    imshow(imx(100:1500,750:2000,:), im_cmap,'InitialMagnification',100)
end
for cii = length(ser_ind)+1:16
    axes(hand(cii));
    wc_name = ['2022_02_16_Wordclouds_120_Topics_LDA\allTopics_Seite_',sprintf('%03d',ser_ind(1)),'.tiff'];
    [imx,im_cmap] = imread(wc_name);
    imshow(imx(100:1500,750:2000,:), im_cmap,'InitialMagnification',100)
end

colormap parula;
elps1 = annotation('ellipse',[.108 .695 .195 .255]);
elps1.Color=cmap(1,:);
elps1.LineWidth=2;
elps2 = annotation('ellipse',[.308 .695 .195 .255]);
elps2.Color=cmap(2,:);
elps2.LineWidth=2;
elps3 = annotation('ellipse',[.50 .695 .195 .255]);
elps3.Color=cmap(3,:);
elps3.LineWidth=2;
elps4 = annotation('ellipse',[.6998 .695 .195 .255]);
elps4.Color=cmap(4,:);
elps4.LineWidth=2;
elps5 = annotation('ellipse',[.108 .375 .195 .255]);
elps5.Color=cmap(5,:);
elps5.LineWidth=2;
elps6 = annotation('ellipse',[.308 .375 .195 .255]);
elps6.Color=cmap(6,:);
elps6.LineWidth=2;
elps7 = annotation('ellipse',[.50 .375 .195 .255]);
elps7.Color=cmap(7,:);
elps7.LineWidth=2;
elps8 = annotation('ellipse',[.6998 .375 .195 .255]);
elps8.Color=cmap(1,:);
elps8.LineWidth=2;
% dim = [.2 .5 .3 .3];
% text = ['inno_grade ',sprintf('%5g',inno_labels(ser_ind(7),10))]
% annotation('textbox',dim,'String',text)

formatOut2='yy-mm-dd-HH-MM';


% figuren = ['results\FR_',datestr(datetime,formatOut2),'_LDA_wordclouds_',int2str(iter),'_',int2str(m),'.pdf'];
% print(figuren,'-dpdf');

end


