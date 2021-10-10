function results = logresults(output,results)
    maxregret=max(output.mean_regret(end));
    results.sums.sum_normalised_regret=results.sums.sum_normalised_regret+output.mean_regret/maxregret; 
    results.sums.sum_normalised_regret2=results.sums.sum_normalised_regret2+(output.mean_regret/maxregret).^2; % normalised regret
    
    results.sums.sum_regret=results.sums.sum_regret+output.mean_regret; 
    results.sums.sum_regret2=results.sums.sum_regret2+(output.mean_regret).^2; % regret
    
    results.sums.sum_ghat_correct=results.sums.sum_ghat_correct+output.ghat_correct; 
    results.sums.sum_ghat_correct2=results.sums.sum_ghat_correct2+(output.ghat_correct).^2; % accuracy
    
    results.count = results.count+1;
    res=find(output.mean_regret/maxregret>=0.8, 1);
    if isempty(res)
        results.metrics.t_converge(results.count) = NaN;
    else
        results.metrics.t_converge(results.count) = find(output.mean_regret/maxregret>=0.8, 1);
    end
    results.metrics.maxregret(results.count) = maxregret;
