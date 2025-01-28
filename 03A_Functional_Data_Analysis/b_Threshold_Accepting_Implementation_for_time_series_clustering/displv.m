function dis_res = displv(vmod,vprio,vtext,vval) 
%     -- subroutine displv -------------------------------------------         
%     This subroutines prints informations depending on the verbose
%     level and the verbose priority of the submitted vtext
                                                                            
%     --- Print the information --------------------------------------         

if (vmod >= vprio)
  if (vval <= -998)
    fprintf('\n %2i --- %25s ---',vprio,vtext);
  else
    fprintf('\n %2i --- %25s  %12.5f',vprio,vtext,vval);
  end
end
vtext='                         ';

%     ------------------------------------- END print ----------------         

dis_res = 0;                                                                             
%     ------------------------------------- END verbose information --         
