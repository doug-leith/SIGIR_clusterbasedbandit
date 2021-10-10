% use a group of items for each distinguisher and optimum arm - every item
% is rated at most once

close all

Tmax=25;
Ntries=100; %Ntries=1000;
Nyms=8; %[4,8,16,32]; % BLC setups to evaluate
%min_ratings=1; % dating, seems weird data
min_ratings=5; % all the rest

train=1; % reload data etc
train_DT=1; %train decision tree
dt=1; % run decision tree
best=1; % run SIGIR cluster-based bandit algo
plot_accuracy=0;
plot_regret=0;
plot_t_converge=0;
TT=[];TT_DT=[]; TT_best=[]; MM=[]; AA=[]; MAE=[]; RMSE=[]; NDCG=[];
count=1;
%figure(100), clf, hold on, xlabel('Group'), ylabel('Convergence Time (steps)'),set(gca,'fontsize',24)
for nyms=Nyms
    % load mean and variances for each group
    G= 1:nyms; % groups
    if train
        % netflix
        cd(strcat('netflix_full_',string(nyms),'nyms'))
        num = readtable('lam.csv');  num=table2array(num);
        sigma = readtable('Rvar.csv');  sigma=table2array(sigma); 
        mu = readtable('rtilde.csv');  mu=table2array(mu);  % item losses mu_g(v) = row g, col v, same sigma_g(v)
        sigma=sigma+0.5*sqrt(log(1/0.2)./num);
        mu=-mu;
 
        % jester
%         cd(strcat('jester_',string(nyms),'nyms'))
%         load('lam.mat'); num=double(lam);
%         load('Rvar.mat'); sigma=Rvar;
%         load('rtilde.mat'); mu=rtilde;
%         mu=-mu;
%         sigma=sigma+0.5*sqrt(log(1/0.2)./(num+0.01));

        %goodreads10k
%         cd(strcat('goodreads_',string(nyms),'nyms'))
%         load('lam.mat'); num=double(lam);
%         load('Rvar.mat'); sigma=Rvar;
%         load('rtilde.mat'); mu=rtilde;
%         mu=-mu;
%         sigma=sigma+0.5*sqrt(log(1/0.2)./num);

%       % if some nyms have hardly any users, remove them
%       % the values here are for netflix and goodreads data (16 nyms), not used with
%       jester data
        G=[]; [num_nyms,~]=size(mu); % num_nyms might not be same as nyms
        for g=1:num_nyms, if length(find(num(g,:)>5))>100, G=[G,g]; end; end
        G=G(1:min(nyms,length(G))); % truncate if needed
        num=num(G,:); mu=mu(G,:); sigma=sigma(G,:);
        G=1:length(G);
        
        cd ..        
    end
    
    num_items=Tmax+1; % number of items in each distinguisher group/arm (same value is also used for items in each optimum arm)
    num_vstar_items=Tmax+1; % number of items for each optimum arm
    if train
        % distinguisher d_{g,h} = row g, col h.  vstar is set of best arms for each group
        % items is set of distinguisher and optimal items, gammagh(g,h,v)
        % the gamma value for item v and groups g,h.  only entries for
        % v \in items (NaN for other items)
        [d,vstar,items,gammagh]= finddistinguishers2(mu,sigma,num,G,num_items,num_vstar_items,min_ratings);
        
        % use optimal items as distinguishers (to provide a baseline for
        % comparison)
        d_best=NaN(length(G),length(G),num_items);
        for g = G
            for h = G
                if (g==h), continue; end
                d_best(g,h,:)=vstar(g,1:num_items);
            end
        end
            
    end
  
    if dt && train_DT % train decision tree
        [tree, tree_items] = trainDecisionTree(G,mu,sigma,num,min_ratings);
    end
    
    for gg=G
        results=initResults(Tmax,Ntries,gg,nyms); % holds results for new algo
        results_DT=initResults(Tmax,Ntries,gg,nyms);  % holds results for decision tree
        results_best=initResults(Tmax,Ntries,gg,nyms); % holds results for bandit based only on best arms
        sum_err = init_sum_err(); sum_err_DT = init_sum_err(); sum_err_best = init_sum_err(); sum_err_true = init_sum_err();
        for tries=1:Ntries
            % generate user item ratings
            Ruser=min(0,sqrt(sigma(gg,:)).*randn(1,length(sigma))+mu(gg,:)); % truncate ratings to be >=0.  is this ok?
            Ruser_ii=find(min(num)>=5);
            sum_err_true = calc_err(Ruser,Ruser_ii,mu,gg,gg,sum_err_true);
            
            % run new algo
            [ghat,output,LL] = newAlgo(Ruser,gg,G,mu,d,vstar,gammagh,items,Tmax);
            results = logresults(output,results); 
            sum_err = calc_err(Ruser,Ruser_ii,mu,gg,ghat,sum_err);
            
            if best
                %run new algo using best (highest rewards) arms as distinguishers
                [ghat_best,output_best,LL_best] = newAlgo(Ruser,gg,G,mu,d_best,vstar,gammagh,items,Tmax);
                results_best = logresults(output_best,results_best); 
                sum_err_best = calc_err(Ruser,Ruser_ii,mu,gg,ghat_best,sum_err_best);
            end
                        
            if dt % run decision tree
                [ghat_DT,output_DT,LL_DT] = runDecisionTree(Ruser,gg,G,mu,tree,tree_items,vstar,Tmax);
                results_DT = logresults(output_DT,results_DT); 
                sum_err_DT = calc_err(Ruser,Ruser_ii,mu,gg,ghat_DT,sum_err_DT);
            end
        end
        % collect aggregate stats
        stats = calc_stats(results);
        stats_DT = calc_stats(results_DT);
        stats_best = calc_stats(results_best);
        stats_all(count).stats=stats; stats_all(count).gg=gg; stats_all(count).nyms=nyms;
        TT=[TT;gg,nyms,mean(results.metrics.t_converge),std(results.metrics.t_converge),prctile(results.metrics.t_converge,[50,10,20,80,90])];
        TT_DT=[TT_DT;gg,nyms,mean(results_DT.metrics.t_converge),std(results_DT.metrics.t_converge),prctile(results_DT.metrics.t_converge,[50,10,20,80,90])];
        TT_best=[TT_best;gg,nyms,mean(results_best.metrics.t_converge),std(results_best.metrics.t_converge),prctile(results_best.metrics.t_converge,[50,10,20,80,90])];
        MM=[MM;gg,nyms,mean(results.metrics.maxregret),mean(results_DT.metrics.maxregret),mean(results_best.metrics.maxregret)];
        AA=[AA;gg,nyms,stats.accuracy_mean(end),stats_DT.accuracy_mean(end),stats_best.accuracy_mean(end)];
        MAE=[MAE;gg,nyms,sum_err_true.mae/Ntries,sum_err.mae/Ntries,sum_err_DT.mae/Ntries,sum_err_best.mae/Ntries];
        RMSE=[RMSE;gg,nyms,sum_err_true.rmse/Ntries,sum_err.rmse/Ntries,sum_err_DT.rmse/Ntries,sum_err_best.rmse/Ntries];
        %NDCG=[NDCG;gg,nyms,sum_err_true.ndcg/Ntries,sum_err.ndcg/Ntries,sum_err_DT.ndcg/Ntries,sum_err_best.ndcg/Ntries];
        fprintf("nyms %g, group %g, accuracy newalgo=%3.2f/decision tree=%3.2f/newalgo-best=%3.2f\n",nyms,gg,stats.accuracy_mean(end),stats_DT.accuracy_mean(end),stats_best.accuracy_mean(end))
        
        if plot_regret
            % plot regret vs time for current group
            figure(gg), hold off;  clf
            errorbar(stats.regret_mean,stats.regret_std)
            hold on
            errorbar(stats_DT.regret_mean,stats_DT.regret_std,'--')
            errorbar(stats_best.regret_mean,stats_best.regret_std,'o-')
            title(sprintf(" group %g, nyms %g",gg, nyms))
            xlabel('iteration'),ylabel('regret')
            set(gca,'fontsize',24)
        end
        if plot_accuracy
            % plot accuracy vs time for current group
            figure(length(G)+gg), hold off; clf
            errorbar(stats.accuracy_mean,stats.accuracy_std)
            hold on
            errorbar(stats_DT.accuracy_mean,stats_DT.accuracy_std,'--')
            errorbar(stats_best.accuracy_mean,stats_best.accuracy_std,'o-')
            title(sprintf(" group %g, nyms %g",gg, nyms)),xlabel('iteration'),ylabel('accuracy')
            set(gca,'fontsize',24)
        end
    end
    %figure(100)
    %ii=find(TT(:,2)==nyms);
    %errorbar(TT(ii,1),TT(ii,6),TT(ii,6)-TT(ii,7),TT(ii,8)-TT(ii,6))
    %boxplot(TT(ii,3:end),'symbol','') % don't plot outliers
end

% mean (across groups) accuracy vs #nyms and algo
fprintf("Mean accuracy:\n")
fprintf("%5s: ", "CB");
%Nyms=[4     8    16 32];
for nym=Nyms,
    ii=find(AA(:,2)==nym);
    fprintf("%3.2f ",mean(AA(ii,3))) % new algo
end
fprintf("\n%5s: ", "DT");
for nym=Nyms,
    ii=find(AA(:,2)==nym);
    fprintf("%3.2f ",mean(AA(ii,4))) %decision tree
end
fprintf("\n%5s: ","CB-");
for nym=Nyms,
    ii=find(AA(:,2)==nym);
    fprintf("%3.2f ",mean(AA(ii,5))) % new algo, no exploration
end
fprintf("\n");
% mean (across groups) convergence time vs #nyms and algo
fprintf("Mean convergence time:\n")
fprintf("%5s: ", "CB");
for nym=Nyms,
    ii=find(TT(:,2)==nym);
    fprintf("%3.2f ",mean(TT(ii,3))) % new algo
end
fprintf("\n%5s: ", "DT");
for nym=Nyms,
    ii=find(TT_DT(:,2)==nym);
    fprintf("%3.2f ",mean(TT_DT(ii,3))) % decision tree
end
fprintf("\n%5s: ","CB-");
for nym=Nyms,
    ii=find(TT_best(:,2)==nym);
    fprintf("%3.2f ",mean(TT_best(ii,3))) % new algo, no exploration
end
fprintf("\n");
fprintf("Mean max regret:\n")
fprintf("%5s: ", "CB");
%Nyms=[4     8    16 32];
for nym=Nyms,
    ii=find(MM(:,2)==nym);
    fprintf("%3.2f ",mean(MM(ii,3))) % new algo
end
fprintf("\n%5s: ", "DT");
for nym=Nyms,
    ii=find(MM(:,2)==nym);
    fprintf("%3.2f ",mean(MM(ii,4))) %decision tree
end
fprintf("\n%5s: ","CB-");
for nym=Nyms,
    ii=find(MM(:,2)==nym);
    fprintf("%3.2f ",mean(MM(ii,5))) % new algo, no exploration
end
fprintf("\n");


if plot_t_converge,
    %plot convergence time 
    figure(101), clf, hold on, set(gca,'fontsize',24), xlabel('Group'), ylabel('Convergence Time (steps)')
    for nym=Nyms, %[8,16,32,64,128]
        ii=find(TT(:,2)==nym);
        data=TT(ii,3:end); p50=prctile(data',50);p25=prctile(data',25);p75=prctile(data',75);
        [dummy,I]=sort(p50,'descend' );
        errorbar(1:nym,p50(I),p50(I)-p25(I),p75(I)-p50(I),'LineWidth',3)
        hold on
        ii=find(TT_DT(:,2)==nyms);
        data=TT_DT(ii,3:end); p50=prctile(data',50);p25=prctile(data',25);p75=prctile(data',75);
        [dummy,I]=sort(p50,'descend' );
        errorbar(1:nym,p50(I),p50(I)-p25(I),p75(I)-p50(I),'--','LineWidth',3)
        %boxplot(TT(ii,3:end)','symbol','')
    end
    legend('Cluster Bandit','Decision tree')
    %legend('8 groups','16 groups','32 groups','64 groups','128 groups')
end
figure(102)
% bar(AA(:,1),AA(:,3:5),'grouped')
% xlabel('Group'), ylabel('Accuracy'),set(gca,'fontsize',24)
% legend('Cluster Bandit','Decision tree','Cluster Bandit (no exploration)')
bar(AA(:,1),AA(:,3:4),1,'grouped')
xlabel('Group'), ylabel('Accuracy'),set(gca,'fontsize',24)
legend('Cluster Bandit','Decision tree')

%figure(103)
% bar(MAE(:,1),MAE(:,3:6),'grouped')
% xlabel('Group'), ylabel('MAE'),set(gca,'fontsize',24)
% figure(104)
% bar(NDCG(:,1),NDCG(:,3:6),'grouped')
% xlabel('Group'), ylabel('nDCG@25'),set(gca,'fontsize',24)
%mean(MAE(:,3:6))
%mean(RMSE(:,3:6))
%mean(NDCG(:,3:6))
return

