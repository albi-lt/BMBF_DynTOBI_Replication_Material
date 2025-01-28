function objf = objf_Clust_TA(m,n,indexc,fmod,dtm)
%     -- subroutine objf ---------------------------------------------
%     This subroutine calculates objective function for fmod=0,..,3

switch fmod
    case 0 % Summe der Abstände im Cluster geteilt durch Anzahl der Abstände
        objf = 0;
        for k =1:m
          ser_ind = find(indexc==k);
          if length(ser_ind) > 1
             % test = sum(sum(dtm(ser_ind,ser_ind)))/(length(ser_ind)*(length(ser_ind)-1))
             objf = objf + sum(sum(dtm(ser_ind,ser_ind)))/(length(ser_ind)*(length(ser_ind)-1));           
          end
        end
     case 1 % Summe der Abstände zum Prototypen (also kleinste der möglichen Summen) geteilt durch Anzahl der Abstände
         objf = 0;
         for k=1:m
             ser_ind = find(indexc==k);
             if length(ser_ind) > 0
                 objf = objf + min(sum(dtm(ser_ind,ser_ind))); %/length(ser_ind);
             end
         end
%         objf = -trace(inv(M1));
%     case 2 % E-opt
%         objf = -min(eig(M1));
%     case 3 % MS-opt
%         objf = -trace(M1^2);
    otherwise
        objf = NaN
end

%     ------------------------------------- END objective function ---         
                                                                               
