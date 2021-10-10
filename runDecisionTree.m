function [ghat,output,LL_DT] = runDecisionTree(Ruser,gg,~,mu,tree,tree_items,vstar,Tmax)

    [ghat,posterior,node,cnum]=predict(tree,Ruser(tree_items));
    % unfortunately nodeVariableRange sorts decision variables, so its no
    % good.
    %path=tree.nodeVariableRange(node);
    %ppath=str2double(split(fieldnames(path),'x')); ppath=ppath(:,2)'; % the sequence of items queried by the decision tree

    % walk backwards from leaf through the decision tree to get items rated and the predictions at each decision point on tree
    preds=[]; items_rated=[];
    prev=node;
    while prev>0
        preds=[str2double(tree.NodeClass(prev));preds]; % predicted class at this branch
        items_rated=[tree.CutPredictorIndex(prev);items_rated]; % item branch decision is based upon
        prev = tree.Parent(prev);
    end
    % line up items rated and corresponding predictions
    % (prediction is available after item is rated so there's
    % an off by one mismatch)
    preds=preds(2:end); items_rated=items_rated(1:end-1);

    % it's possible that the same item is used at >1 branch,
    % but wrt asking for rating we're only interested in the
    % first time its used.  we do need to take account of later
    % uses when calculating predictions though
    ppreds=[]; already_rated=[];
    for i=1:length(preds)
        if ismember(items_rated(i),already_rated)
            ppreds(end)=preds(i);
            continue;
        end
        ppreds=[ppreds;preds(i)];
        already_rated=[already_rated;items_rated(i)];
    end
    items_rated=already_rated;  % with duplicates removed
    if ghat ~= ppreds(end)
        % ghat returned by predict() call is different from
        % our analysis by walking decision tree.  this can
        % happen when there are two or more equally likely classes and
        % NodeClass() returns a different choice from predict()
        % (but both are equally likely).  let's break ties
        % randomly (nb: predict() and NodeClass() seem
        % non-random)
        if posterior(ghat) ~= posterior(ppreds(end))
            fprintf("problem!  ghat=%d, ppreds(end)=%d, posterior=%3.2f/%3.2f\n",ghat,ppreds(end),posterior(ghat),posterior(ppreds(end)));
            return % bail, this is a proper error
        end
        maxprob=max(posterior); maxi=find(posterior==maxprob);
        ghat2=maxi(randi(length(maxi)));
        %fprintf("ghat=%d, ppreds(end)=%d, picking %d\n",ghat,ppreds(end),ghat2);
        ppreds(end)=ghat2; ghat=ghat2;
    end

    YY_DT=[]; LL_DT=[];
    [~,num_vstar_items]=size(vstar);
    num_vstar_rated=0; 
    I=zeros(1,length(mu)); % keep track of which items have been rated
    %                 for p=ppath
    %                     l=tree_items(p); % item rated
    %                     I(l)=I(l)+1;
    %                     X = Ruser(l); % pull arm
    %                     nn=min(num_vstar_rated+1,num_vstar_items); num_vstar_rated=nn;
    %                     Xstar = Ruser(vstar(gg,nn));
    %                     regret = regret + X-Xstar;
    %                     regret2 = regret2 + mu(gg,l)-mu(gg,vstar(gg,nn));
    %                     YY_DT=[YY_DT,regret2];
    %                     LL_DT=[LL_DT;l,vstar(gg,nn),X,Xstar,mu(gg,l),mu(gg,vstar(gg,nn))];
    %                 end
    % vectorize above loop for a bit of speed up ...
    if (length(items_rated)>Tmax) % truncate if run out of time
        items_rated=items_rated(1:Tmax);
        ppreds=ppreds(1:Tmax);
    end
    ll=tree_items(items_rated);
    X=Ruser(ll); I(ll)=1;
    Xstar= Ruser(vstar(gg,1:length(ll)));
    nn=min(num_vstar_rated+length(ll)+1,num_vstar_items); num_vstar_rated=nn;
    regret = sum(X-Xstar);
    regret2 = sum(mu(gg,ll)-mu(gg,vstar(gg,1:length(ll))));
    YY_DT=[YY_DT;cumsum(X-Xstar)', cumsum(mu(gg,ll)-mu(gg,vstar(gg,1:length(ll))))', ppreds==gg];
    LL_DT=[LL_DT;ll',vstar(gg,1:length(ll))',X',Xstar',mu(gg,ll)',mu(gg,vstar(gg,1:length(ll)))'];

    % tree has been walked, now rate best items from estimated
    % group ghat
    num_vstar_rated2=0;
    for t=1:(Tmax-length(items_rated))
        nn=min(num_vstar_rated+1,num_vstar_items); num_vstar_rated=nn;
        nn2=min(num_vstar_rated2+1,num_vstar_items); num_vstar_rated2=nn2;
        l=vstar(ghat,nn2);
        while I(l)>0 && num_vstar_rated2<num_vstar_items
            % item already rated
            nn2=min(num_vstar_rated2+1,num_vstar_items); num_vstar_rated2=nn2;
            l=vstar(ghat,nn2);
        end
        I(l)=I(l)+1;
        X=Ruser(l);
        Xstar = Ruser(vstar(gg,nn));
        regret=regret+X-Xstar;
        regret2 = regret2 + mu(gg,l)-mu(gg,vstar(gg,nn));
        YY_DT=[YY_DT;regret,regret2,ghat==gg];
        LL_DT=[LL_DT;l,vstar(gg,nn),X,Xstar,mu(gg,l),mu(gg,vstar(gg,nn))];
    end
    ii=find(I>1);
    if (~isempty(ii)) % sanity check: we should only have rated each item at most once
        for j=ii
            fprintf("DT: problem! item %d rated %d times ",j,I(j));
        end
        fprintf("\n");
    end
    output=table(YY_DT(:,1),YY_DT(:,2),YY_DT(:,3),length(items_rated)*ones(length(YY_DT),1),'VariableNames',{'regret','mean_regret','ghat_correct','t_converge'});