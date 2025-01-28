function [best,best1,objf_mean,objf_var,Dbest] = Clust_TA_1(m,n,runs,nn,un,iter,dtm);

%  ----------------------------------------------------------------
%  I                                                              I
%  I  Author: P.Winker                                            I
%  I                                                              I
%  I  Last Update: 28-09-21  12.00 h                              I
%  I                                                              I
%  ----------------------------------------------------------------

%     --- start clock ------------------------------------------------
% clear, close all, flops(0), tic;

%     ------------------------------------- END start clock ----------

%     --- initialize discb = 999999 ----------------------------------
objfb = 999999;
% ortflag = 0;
%     ------------------------------------- END init discb = 999999 --

nvval=-999;
vtext='Clust_TA - run';
displv(0,0,vtext,nvval);

%     --- setting parameters -----------------------------------------
m =  m;              % number of cluster
n =  n;              % number of series for clustering
nn = nn;             % next neighbours
un = un;             % distance of next neighbour cluster

%     --- Selections for optimization algorithm (outer loop) ----------
wmod   = 0;           % 0: uniform       1: normal distrib. of NN
tmod   = 0;           % 0: step-         1: density-method (not avail.)
tfakl  = 0.50000;      % threshold
tfaku  = 0.70000;      % factor range
%tsteps = 6;           % steps
tfaku  = 0.50000;      % factor range
tsteps = 0;           % steps
rfakl  = 1.0000;      % round
rfaku  = 1.0000;      % factor range
rsteps = 0;           % steps
seedl  = 160;         % starting value range
seedu  = 159 + runs;         % for random number generator
iter   = iter;        % total number of iterations
%     --- General parameter settings ------------------------------------
fmod  = 1;            % 0: Summe der paarweisen Distanzen, 1: Prototyp
vmod  = 0;            % 0: silent        >0: verbose level
vmod  = 1;            % 0: silent        >0: verbose level
zmod  = 0;            % 0: without       1: with cpu-time control

vtext='parameters loaded        ';
displv(vmod,1,vtext,nvval);
%     ------------------------------------- END set parameters -------


% Sonderoutput für Folien:
filer = ['results\Clust_TA_Verteilung_',mat2str(m),'_',mat2str(n),'_',mat2str(iter/1000000),'.data'];
f18 = fopen(filer,'w');

%     --- open files for results -------------------------------------
filer = ['results\Clust_TA_1_',mat2str(m),'_',mat2str(n),'_',mat2str(iter/1000000),'.data'];

f16 = fopen(filer,'w');
fprintf(f16,'\n--- Used parameters -------------------------------------');
fprintf(f16,'\n');

vtext='result file open         ';
displv(vmod,1,vtext,nvval);

%     ***** verbose = 3 ***********************************************
if (vmod >= 3)
    f17 = fopen('results\Clust_TA_1.log','w');
    vtext='log file open            ';
    displv(vmod,3,vtext,nvval);
end
%     *****************************************************************

%     ------------------------------------- result files open --------
fprintf(f16,'\n m  = %10i',m);
fprintf(f16,'\n N  = %10i',n);
fprintf(f16,'\n ud = %10i',nn);

fprintf(f16,'\n wmod = %10i',wmod);
fprintf(f16,'\n tmod = %10i',tmod);

fprintf(f16,'\n Threshold Factors %7.4f -- %7.4f in %7i Steps',tfakl,tfaku,tsteps);
fprintf(f16,'\n Round Factors     %7.4f -- %7.4f in %7i Steps',rfakl,rfaku,rsteps);
fprintf(f16,'\n Seeds  %7i -- %7i',seedl,seedu);
fprintf(f16,'\n Iterations: %10i',iter);

fprintf(f16,'\n fmod  = %10i',fmod);
fprintf(f16,'\n vmod  = %10i',vmod);
fprintf(f16,'\n zmod  = %10i \n',zmod);
%     ------------------------------------- END open files for... ----


best = 0;
best1 = 0;
Dbest = zeros(m,m);

%     --- tfak loop --------------------------------------------------
for lt=0:tsteps
    if (tsteps > 0)
        tfak=tfakl+lt*(tfaku-tfakl)/tsteps;
    else
        tfak=tfaku;
    end
    
    if (tfak < 1)
        blowup=1;
    else
        blowup=tfak;
        tfak=0.95;
    end
    
   % vtext=' blowup =                ';
   % displv(vmod,0,vtext,blowup);
    
    %     --- rfak loop --------------------------------------------------
    for lr=0:rsteps
        if (rsteps > 0)
            rfak=rfakl+lr*(rfaku-rfakl)/rsteps;
        else
            rfak=rfaku;
        end
        
        %     --- seeds loop -------------------------------------------------
        for ls=seedl:seedu
            %     --- initialize random number generator --
            incr = 1;
            incr2 = 2;
            S = ls;
            if ((wmod == 0))
                rand('state',S) %, disp('**** SEED FIXED ****');
            else
                rand('state',S), disp('**** SEED FIXED ****');
            end
            %     -----------------------------------------
            
            %      call displt(zmod,'before generation of threshold sequence',timem0)
            
            %     --- generation of threshold sequence ---------------------------
            vtext='Gen. of Threshold seq.   ';
            displv(vmod,1,vtext,nvval);
            displv(vmod,1,' tfak =                  ',tfak);
            displv(vmod,1,' rfak =                  ',rfak);
            displv(vmod,1,' seed =                  ',ls*1d0);
            
            %     --- parameter test ---
            if (iter > 500000000)
                error = 'main - threshold - too many iterations  ';
                ifail = 10;
                help(error,ifail);
            end
            if (iter < 100)
                error = 'main - threshold - not enough iterations';
                ifail = 11;
                help(error,ifail);
                iter = 100;
            end
            
            vtext='Gen. of jump distribution';
            displv(vmod,1,vtext,nvval);
            
            rounds = floor(sqrt(iter)*rfak);
            rounds_k=rounds-floor(rounds*(1-tfak))+1;
            its = floor((iter - rounds)/rounds_k);
            iterc=rounds_k*its+rounds;
            if ((iterc < (iter*0.95)) | (iterc > (iter*1.05)))
                fprintf('\n Deviation in total number of iterations > 5 %!');
                fprintf('\n Values: %10i %10i ',iter,iterc);
            end
            iterc=iter-iterc;
            
            displv(vmod,1,'Empirical distrib.:      ',rounds*1d0);
            displv(vmod,1,'Rounds:                  ',rounds_k*1d0);
            displv(vmod,1,'Iter. per round:         ',its*1d0);
            displv(vmod,1,'Iter. corrected:         ',iterc*1d0);
            
            for k=1:rounds
                
                %     --- generate randomly matrix XR ---
                % select randomly N rows out of XF with DFr rows;
                indexc = randsample(m,n,true);
                % Xc = XF(indexc,:);
                
                objfc = objf_Clust_TA(m,n,indexc,fmod,dtm);
                
                %     --- generate randomly neighbour to x ---
                indext = indexc;
                %                 Xt = Xc;
                for j=1:nn
                    index_row = random('unid',n);
                    rold=indext(index_row);
                    rnew=rold + random('unid',2*un+1)-un-1;
                    rnew=max(min(m,rnew),1);
                    % replace if rnew not in Xc
                    %                     if isempty(find(indexc==rnew))
                    %                         Xt(index_row,:) = XF(rnew,:);
                    indext(index_row) = rnew;
                    %                     end
                end
                
                objft = objf_Clust_TA(m,n,indext,fmod,dtm);
                
                displv(vmod,5,' thres(k) =              ',(objft-objfc));
                
                thres(k)=abs(objft-objfc)*blowup;
                steps(k)=its;
                
            end
            
            if (iterc > 0)
                steps(rounds)=steps(rounds)+iterc;
            end
            
            thres = -sort(-thres);
            
            vtext='Emp. jump distrib. gener.';
            displv(vmod,1,vtext,nvval);
            
            if ((vmod >= 2) & (vmod ~= 5))
                for i=1:rounds
                    fprintf('\n %5i %8.5f',i,thres(i));
                end
            end
            
            %     ------------------------------------- END generation of...------
            %
            %         end
            %     end
            % end
            
            
            %      call displt(zmod,'after generation of threshold sequence',timem0)
            
            
            %     --- initialize X -----------------------------------------------
            % select randomly N rows out of XF with DFr rows;
            indexc = randsample(m,n,true);
            
            objfc = objf_Clust_TA(m,n,indexc,fmod,dtm);
            
            %Xopt = Xc;
            indexopt=indexc;
            if (objfb == 999999)
                %Xbest = Xopt;
                indexbest = indexopt;
            end
            
            vtext='start matrix generated   ';
            displv(vmod,1,vtext,nvval);
            
            %     ------------------------------------- END initialization of X --
            
            %     --- Initialization ---------------------------------------------
            objfc = objf_Clust_TA(m,n,indexc,fmod,dtm);
            objfo = objfc;
            
            vtext='Initialization successful';
            displv(vmod,1,vtext,nvval);
            vtext='starting value:          ';
            displv(vmod,1,vtext,objfc);
            
            if ((vmod >= 1) & (lt == 0) & (lr == 0) & (ls == seedl))
                timer=toc/rounds;
                times=floor(timer*(rsteps+1)*(tsteps+1)*(seedu-seedl+1)*iter);
                timeh=floor(times/3600);
                timem=floor((times-(timeh*3600))/60);
                times=(times-(timeh*3600)-(timem*60));
                fprintf('\n --- Approximative remaining computing time -------');
                fprintf('\n (estimated without fast updating!)');
                fprintf('\n  %5i h %3i m %3i s',timeh,timem,times);
            end
            
            %     ------------------------------------- END Initialization -------
            
            %      call displt(zmod,'initialization',timem0)
            
            %     --- Optimization -----------------------------------------------
            
            vtext='Optimization running     ';
            displv(vmod,1,vtext,nvval);
            
            for k=(floor(rounds*(1d0-tfak))):rounds
                random_1 = random('unid',n,steps(k),nn);
                random_2 = random('unid',2*un+1,steps(k),nn);
                for l=1:steps(k)
                    indext = indexc;
                    % Xt = Xc;
                    for j=1:nn
                        % index_row = random('unid',n);
                        index_row = random_1(l,j);
                        rold=indext(index_row);
                        % rnew=rold + random('unid',2*un+1)-un-1;
                        rnew = rold + random_2(l,j)-un-1;
                        rnew=max(min(m,rnew),1);
                        indext(index_row) = rnew;
                    end
                    
                    objft = objf_Clust_TA(m,n,indext,fmod,dtm);
                    
                    if isnan(objft)
                       fprintf('\n NAN!') 
                    end
                    %     ***** verbose = 3 ***********************************************
                    if ((vmod >= 3) & (vmod ~= 20))
                        fprintf(f17,'\n %8.5f %8.5f ',objft,objfc);
                    end
                    
                    %     *****************************************************************
                    
                    if (objft < (objfc+thres(k)))
                        indexc = indext;
                        
                        if (objft < objfo)
                            objfo = objft;
                            indexopt = indext;
                            %     ***** verbose > 0 ***********************************************
                            if (vmod >= 2)
                                fprintf('\n --- improvement achieved -------------------------');
                                fprintf('\n %8.5f %8.5f %8.5f %8.5f %8.5f',tfak, rfak, ls, objfc, objft);
                            end
                            %     *****************************************************************
                            
                        end
                        objfc=objft;
                    end % nn
                end % steps
            end % rounds
            
            
            vtext='Optimization ended       ';
            displv(vmod,1,vtext,nvval);
            
            %     ------------------------------------- END optimization ---------
            fprintf(f18,'\n %8.5g %16.5g %16.5g',tfak,objfo,objfc);

            
            %     --- Saving results ---------------------------------------------
            
            %     --- only one seed ---
            if (seedl == seedu)
                %       --- only one rfak and tfak ---
                if ((tsteps == 0) & (rsteps == 0))
                    fprintf(f16,'\n\n--- Values of objective function ----------------');
                    fprintf(f16,'\n best value: %16.5g  last value: %16.5g',objfo,objfc);
                    fprintf(f16,'\n');
                    objfb=objfo;
                    %Xbest = Xopt;
                    indexbest = indexopt;
                    
                    %       --- multiple rfak or tfak ---
                else
                    if ((lt == 0) & (lr == 0))
                        fprintf(f16,'\n\n  tfak    rfak   seed         best value          last value');
                        objfb=objfo;
                        Xbest = Xopt;
                        indexbest = indexopt;
                    end
                    fprintf(f16,'\n %7.4f %7.4f %4i %16.7g %16.7g',tfak,rfak,ls,objfo,objfc);
                    if (objfo < objfb)
                        objfb=objfo;
                        %Xbest = Xopt;
                        indexbest = indexopt;
                    end
                end
                
                %     --- multiple seeds ---
            else
                %       --- only one rfak and tfak ---
                if ((tsteps == 0) & (rsteps == 0))
                    %         --- first of multiple seeds ---
                    if (ls == seedl)
                        fprintf(f16,'\n\n--- Values of objective function ------------------');
                        %write(16,3120)
                        fprintf(f16,'\n %4i %16.5g %16.5g',ls, objfo,objfc);
                        objfb=objfo;
                        %Xbest = Xopt;
                        indexbest = indexopt;
                        objfm=objfo;
                        objfr(1)=objfo;
                        oamean=0;
                        oapcbe=0;
                        
                        %         --- later seed ---
                    else
                        fprintf(f16,'\n %4i %16.5g %16.5g',ls, objfo,objfc);
                        objfm=objfm+objfo;
                        objfr(ls-seedl+1)=objfo;
                        if (objfo < objfb)
                            objfb=objfo;
                            %Xbest = Xopt;
                            indexbest = indexopt;
                        end
                    end
                    
                    %         --- last seed ---
                    if (ls == seedu)
                        objfm=objfm/(seedu-seedl+1);
                        objfv=0;
                        objfbe=0;
                        for i=1:(seedu-seedl+1)
                            objfv=objfv+(objfr(i)-objfm)^2;
                            if (objfr(i) == objfb)
                                objfbe=objfbe+1;
                            end
                        end
                        objfv=sqrt(objfv/(seedu-seedl+1));
                        fprintf(f16,'\n\n mean value         std. dev.           optimum          pc.');
                        fprintf(f16,'\n %16.7g   %16.7g   %16.7g   %7.3f',objfm,objfv,objfb, (100*objfbe/((seedu-seedl+1))));
                        fprintf(f16,'\n');
                        oamean=oamean+objfm;
                        oapcbe=oapcbe+100*objfbe/(seedu-seedl+1);
                    end
                    %       --- multiple rfak or tfak ---
                else
                    if ((lt == 0) & (lr == 0) & (ls == seedl))
                        fprintf(f16,'\n\n--- Values of objective function -------------------------------');
                        fprintf(f16,'\n\n  tfak    rfak    mean value         std. dev.           optimum          percent');
                        oamean=0;
                        oapcbe=0;
                        objfb=objfo;
                        %Xbest = Xopt;
                        indexbest = indexopt;
                    end
                    %         --- first of multiple seeds ---
                    if (ls == seedl)
                        objfb2=objfo;
                        objfm=0;
                    end
                    
                    %         --- later seed ---
                    objfm=objfm+objfo;
                    objfr(ls-seedl+1)=objfo;
                    if (objfo < objfb2)
                        objfb2=objfo;
                    end
                    if (objfo < objfb)
                        objfb=objfo;
                        oapcbe=0;
                        %Xbest = Xopt;
                        indexbest = indexopt;
                    end
                    
                    %         --- last seed ---
                    if (ls == seedu)
                        objfm=objfm/(seedu-seedl+1);
                        objfv=0;
                        objfbe=0;
                        for i=1:(seedu-seedl+1)
                            objfv=objfv+(objfr(i)-objfm)^2;
                            if (objfr(i) == objfb)
                                objfbe=objfbe+1;
                            end
                        end
                        objfv=sqrt(objfv/(seedu-seedl+1));
                        fprintf(f16,'\n %7.4f %7.4f %16.7g  %16.7g  %16.7g  %7.3f',tfak,rfak,-objfm,objfv,-objfb2,(100*objfbe/(seedu-seedl+1)));
                        oamean=oamean+objfm;
                        oapcbe=oapcbe+100*objfbe/(seedu-seedl+1);
                    end
                    
                end
                
                
                %     ***** verbose > 0 ***********************************************
                if (vmod > 0)
                    fprintf('\n %7.3f %7.3f %4i %16.7g %16.7g %16.7g',tfak, rfak, ls, objfo, objfc, objfb);
                end
                %     *****************************************************************
                
                %             end
                %             %     ------------------------------------- END seed loop ------------
                %         end
                %         %     ------------------------------------- END rfak loop ------------
                %     end
                %     %     ------------------------------------- END tfak loop ------------
            end
            end % seed
        end % rsteps
    end % tsteps
    
    
    fprintf(f16,'\n');
    f16suc = fclose(f16);
    f16suc = fclose(f18);
    
    %     ***** verbose = 3 ***********************************************
    if (vmod >= 3)
        f17suc = fclose(f17);
    end
    %     *****************************************************************
    
    vtext='Clust_TA_1 finished sucessful';
    displv(vmod,1,vtext,nvval);
    
    %     ------------------------------------- END Saving results -------
    
    best = objfb;
    best1 = objfc;
    objf_mean = objfm;
    objf_var = objfv;
    Dbest = indexbest;
    return
