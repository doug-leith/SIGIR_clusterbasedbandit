function results=initResults(Tmax,Ntries,gg,nyms)
    sum_normalised_regret=zeros(Tmax,1); sum_normalised_regret2=zeros(Tmax,1);
    sum_regret=zeros(Tmax,1); sum_regret2=zeros(Tmax,1);
    sum_ghat_correct=zeros(Tmax,1); sum_ghat_correct2=zeros(Tmax,1);
    sum_results=table(sum_normalised_regret,sum_normalised_regret2,sum_regret,sum_regret2,sum_ghat_correct,sum_ghat_correct2);
    
    max_regret=NaN(Ntries,1); t_converge=NaN(Ntries,1);
    metric_results=table(max_regret,t_converge);
    results=struct('metrics',metric_results,'sums',sum_results,'count',0,'gg',gg,'nyms',nyms);