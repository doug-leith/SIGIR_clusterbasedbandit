function sum_err=calc_err(Ruser,Ruser_ii,mu,gg,ghat,sum_err)
    sum_err.mae = sum_err.mae+mean(abs(Ruser(Ruser_ii)-mu(ghat,Ruser_ii)));
    sum_err.rmse = sum_err.rmse+std(abs(Ruser(Ruser_ii)-mu(ghat,Ruser_ii)));
    [~,I]=mink(mu(ghat,Ruser_ii),25);
    pred_relevance = -Ruser(Ruser_ii(I));
    [~,I]=mink(Ruser(Ruser_ii),25);
    y=-Ruser(Ruser_ii(I));
    
%     [~,I]=mink(mu(ghat,Ruser_ii),25);
%     pred_relevance = -mu(gg,Ruser_ii(I));
%     [y,I]=mink(mu(gg,Ruser_ii),25); y=-y;
 
    %sum_err.ndcg = sum_err.ndcg+ndcg(pred_relevance, y);