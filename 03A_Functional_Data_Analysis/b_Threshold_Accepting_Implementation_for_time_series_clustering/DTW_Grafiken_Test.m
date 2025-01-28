% Echtzeit definieren
zeit = 1996+4/12:1/12:2021+10/12;

% load
load('dtw_series_time.mat','dtw_series_time');
load('dtw_series_val.mat','dtw_series_val');
load('dtm.mat','dtm');
load('zrmr.mat','zrmr');

load('results\Dbest_120_0.1_25-01-22_07-59.mat','results_Dbest');  
iter = 10;

%inno_labels = readmatrix('lda_innovation_labelling.csv','FileType','text','NumHeaderLines',1,'Delimiter',',');
%inno_grade = inno_labels(:,10)/8;
inno_labels = readmatrix('lda-labelling2_22_03_29.csv','FileType','text','NumHeaderLines',1,'Delimiter',',');
inno_grade = inno_labels(:,10)>0.5;
%inno_grade = inno_labels(:,10);
%inno_cut_off = 5;
%inno_grade = inno_labels(:,10) >= inno_cut_off;

formatOut='yy-mm-dd_HH-MM';
filer = ['results\',datestr(datetime,formatOut),'_Summary.txt']
f18 = fopen(filer,'w');
fprintf(f18,'\n--- Summary results for Clustering -------------------------------------');
fprintf(f18,'\n');

fprintf('\n Mean innovation grade according to selected definition is %4.2f',mean(inno_grade));
fprintf(f18,'\n Mean innovation grade according to selected definition is %4.2f',mean(inno_grade));
fprintf(f18,'\n');

% Schleife über alle Ergebnisse in results_DBest
%for Di = 1: size(results_Dbest,1)
for Di = 1:1
    Dbest = results_Dbest(Di,:)';
    m = max(Dbest);
    nr_rows = round(sqrt(m));
    nr_cols = ceil(m/nr_rows);
    fprintf(f18,'\n Cluster   Number    Mean                         ');
    fprintf(f18,'\n Number    of Topics Innograd. Prototype Innograd.');
    
% Abbildung "echte" Zeitreihen
figure
for ci = 1:m
    s(ci) = subplot(nr_rows,nr_cols,ci);
    hold on
    indices = find(Dbest == ci);
    for cii = 1:length(indices)
        plot(zeit,zrmr(:,indices(cii)));
    end
end

formatOut2='yy-mm-dd-HH-MM';
title_text = ['LDA\_',datestr(datetime,formatOut2),'\_',int2str(iter(1)),'\_',int2str(m)];
title(s(1),title_text);

 figuren = ['results\',datestr(datetime,formatOut2),'_LDA_real_',int2str(iter(1)),'_',int2str(m),'.pdf'];
 print(figuren,'-dpdf');
 
% Abbildung nach Time Warping
 
figure
for ci = 1:m
     s(ci) = subplot(nr_rows,nr_cols,ci);
     hold on
     ser_ind = find(Dbest == ci);
     prototyp = find(sum(dtm(ser_ind,ser_ind))==min(sum(dtm(ser_ind,ser_ind))));
     indices = find(Dbest == ci);
     topic_inno = 0;
     for cii = 1:length(ser_ind)
         inno_col = inno_grade(ser_ind(cii));
         topic_inno = topic_inno + inno_col;
         if (cii == prototyp(1))
             %plot(zeit,zrmr(:,ser_ind(cii)),'Color',[0 0 1],'LineWidth',inno_col*4+0.1);
              plot(zeit,zrmr(:,ser_ind(cii)),'Color',[1-inno_col inno_col 0],'LineWidth',2);
             fprintf('\n Protoyp for Cluster %3i with %3i Topics is Topic %3i with Innograd %4.2f - mean Innograd:',ci,length(ser_ind),ser_ind(cii)-1,inno_col);
         else
             xv = dtw_series_time(ser_ind(prototyp),ser_ind(cii),:);
             x = xv(:);
             yv = dtw_series_val(ser_ind(prototyp),ser_ind(cii),:);
             y = yv(:);
             plot(zeit(x),zrmr(y,ser_ind(cii)),'Color',[1-inno_col inno_col 0]);
         end
     end
     ax = gca;
    topic_inno_mean = topic_inno/length(ser_ind);
    if topic_inno_mean < 0.5
        ax.Color=topic_inno_mean*[1 1 1]+[0.5 0.5 0.5];
    else
        ax.Color=[1 1 1];
    end
    fprintf(' %4.2f',topic_inno_mean);
    fprintf(f18,'\n %7i %9i %9.2f %9i %9.2f',ci,length(ser_ind),topic_inno/length(ser_ind),ser_ind(prototyp)-1,inno_grade(ser_ind(prototyp)));
 end
 fprintf('\n');
  fprintf(f18,'\n');
 
 
formatOut2='yy-mm-dd-HH-MM';
title_text = ['LDA\_',datestr(datetime,formatOut2),'\_',int2str(iter),'\_',int2str(m)];
title(s(1),title_text);

axf = gcf;
axf.InvertHardcopy='off';

figuren = ['results\',datestr(datetime,formatOut2),'_LDA_dtm_',int2str(iter),'_',int2str(m),'.pdf'];
print(figuren,'-dpdf');

% Abbildung der Word Clouds für die Prototypen
 
figure
[hand,posi] = tight_subplot(nr_rows,nr_cols,[.02 -.2],[.05 .01],[.01 .01]);
for ci = 1:m
    % s(ci) = subplot(nr_rows,nr_cols,ci);
    axes(hand(ci));
    %  hold on
     ser_ind = find(Dbest == ci);
     prototyp = find(sum(dtm(ser_ind,ser_ind))==min(sum(dtm(ser_ind,ser_ind))));
     indices = find(Dbest == ci);
     for cii = 1:length(prototyp)
         % strcat('SomeText', sprintf('%02d',5))
         wc_name = ['2022_02_16_Wordclouds_120_Topics_LDA\allTopics_Seite_',sprintf('%03d',ser_ind(prototyp(cii))),'.tiff'];
         [imx,im_cmap] = imread(wc_name);
         imshow(imx(100:1500,750:2000,:), im_cmap,'InitialMagnification',100)
     end
 end


formatOut2='yy-mm-dd-HH-MM';
title_text = ['LDA\_',datestr(datetime,formatOut2),'\_',int2str(iter),'\_',int2str(m)];
title(s(1),title_text);

figuren = ['results\',datestr(datetime,formatOut2),'_LDA_wordclouds_',int2str(iter),'_',int2str(m),'.pdf'];
print(figuren,'-dpdf');

end

fprintf(f18,'\n');
f18suc = fclose(f18);
