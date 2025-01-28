function [dtw_dist, eukl_dist, dtw_series]  = DTW_calc(z1,z2)

dtwv = inf(length(z1)+1,length(z2)+1);
dtwv(1,1) = 0;

dtw_index = zeros(length(z1),length(z2));

for i=2:length(z1)+1
    for j=2:length(z2)+1
        dist = (z1(i-1) - z2(j-1))^2;
        dtwv(i,j) = dist + min([dtwv(i-1,j),dtwv(i,j-1),dtwv(i-1,j-1)]);
        % Bei Gleichheit wird kleinster Wert vorgezogen
        dtw_index(i-1,j-1) = min(find([dtwv(i-1,j),dtwv(i,j-1),dtwv(i-1,j-1)] == min([dtwv(i-1,j),dtwv(i,j-1),dtwv(i-1,j-1)])));
    end
end

% Construct actual series recursively from end - backtracking?
dtw_series = nan(length(z1),2);
dtw_series(length(z1),2) = length(z1);
dtw_series(length(z1),1) = length(z1);
dtw_series(1,1) = 1;
dtw_series(1,2) = 1;

i = length(z1);
j = length(z2);
while (i > 1) | (j > 1)
    switch dtw_index(i,j)
        case 1 % (in Tabelle nach oben)
            dtw_series(i-1,1)=dtw_series(i,1)-1;
            dtw_series(i-1,2)=dtw_series(i,2);
            i = i - 1;
        case 2 % (in Tabelle nach links)
            dtw_series(i,2)=dtw_series(i,2)-1;
            j = j - 1;
        case 3 % (in Tabelle diagonal)
            dtw_series(i-1,1)=dtw_series(i,1)-1;
            dtw_series(i-1,2)=dtw_series(i,2)-1;
            i = i - 1;
            j = j - 1;
        end
end

eukl_dist = sqrt(sum((z1 - z2).^2));
dtw_dist = sqrt(dtwv(length(z1)+1, length(z2)+1));

%     ------------------------------------- END DTW_cala ---         
                                                                               
