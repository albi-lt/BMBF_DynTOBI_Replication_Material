function [zrmr, dtm, dtw_series_time, dtw_series_val]  = DTW_Berechnung(zrmatrix,n)

number_topics = n;

% Zeitreihen der Topicgewichte einlesen
zrmr = readmatrix(zrmatrix,'FileType','text','NumHeaderLines',1,'Delimiter',',');
zrmr = zrmr(:,2:number_topics+1);

% Berechnung der DTW-Distanzen und Speichern der time-warped
% Zeitreihen für Grafiken

dtm = nan(number_topics,number_topics);
eukl_dtm = nan(number_topics,number_topics);

dtw_series_time = nan(number_topics,number_topics,length(zrmr));
dtw_series_val = nan(number_topics,number_topics,length(zrmr));

for i=1:number_topics
for j=1:number_topics
    [dtm(i,j), eukl_dtm(i,j), dtw_series] = DTW_calc(zrmr(:,i),zrmr(:,j));
    dtw_series_time(i,j,:) = dtw_series(:,1);
    dtw_series_val(i,j,:) = dtw_series(:,2);
end
end

% Speichern
save('zrmr.mat','zrmr');
save('dtm.mat','dtm');
save('dtw_series_time.mat','dtw_series_time');
save('dtw_series_val.mat','dtw_series_val');

% Grafiken
% for i=1:10
%     figure
%     plot(zeit,zrmr(:,i),'g')
%     hold on
%     for j=1:80
%         xv = dtw_series_time(i,j,:);
%         x = xv(:);
%         yv = dtw_series_val(i,j,:);
%         y = yv(:);
%         plot(zeit(x),zrmr(y,j),'b');
% %         if j ~= i
% %             plot(zeit,zrmr(:,j),'r');
% %         end
%     end
% end
