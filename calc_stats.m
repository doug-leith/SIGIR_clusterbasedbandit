function stats = calc_stats(results)
    Ntries =  results.count;
    % normalised regret mean and std dev
    normalisedregret_mean=results.sums.sum_normalised_regret/Ntries; 
    normalisedregret_std=sqrt(max(0,results.sums.sum_normalised_regret2/Ntries-normalisedregret_mean.^2)); 
    
    % regret
    regret_mean=results.sums.sum_regret/Ntries; 
    regret_std=sqrt(max(0,results.sums.sum_regret2/Ntries-regret_mean.^2)); 
    accuracy_mean=results.sums.sum_ghat_correct/Ntries; 
    
    % accuracy
    accuracy_std=sqrt(max(0,results.sums.sum_ghat_correct2/Ntries-accuracy_mean.^2));    
    
    stats=table(normalisedregret_mean,normalisedregret_std,regret_mean,regret_std,accuracy_mean,accuracy_std);