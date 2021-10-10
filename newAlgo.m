function [ghat,output,LL] = newAlgo(Ruser,gg,G,mu,d,vstar,gammagh,items,Tmax)
% Ruser is vector of item ratings for new user, gg is their group
% G is set of groups e.g. 1:8
% d is distinguishers: d(g,h,:) is vector of length num_items that gives
% items associated with distinguisher arm for groups g and h
% vstar is optimal arms: vstar(g,:) is vector of length num_vstar_items that
% gives items associated with optimal arm for group g
% gammagh: gamma(g,h,v) is gamma(g,h) value for item v
% items is the vector of item indices corresponding to elements of gamma(g,h,:)

%TODO: 
%if we have population info on how likely a user is to belong to each
%group can we use that to skew the startup towards more probable groups?
%
% how to better select criteria for exiting exploration phase?
B=5;
C=0.5;

[~,~,num_items]=size(d);
[~,num_vstar_items]=size(vstar);

XX=[]; regret=0;  regret2=0;
R=zeros(length(G),length(G));
%history_reward=[]; history_item=[];
Sigma=zeros(length(G),length(G));
Dg_sum=zeros(length(G),length(G));
I=zeros(1,length(mu)); % count of #times each arm pulled
num_d_rated=zeros(length(G),length(G));  % keep track of which #items in each distinguisher that user has rated
num_vstar_rated=zeros(length(G)); % same for optimum arms
num_vstar_rated2=0;
IIhat=[]; LL=[];
for t=1:Tmax
    Rmin = zeros(1,length(G));
    for g = G
        ii = find(G ~= g);
        Rmin(g)=min(R(g,ii));
    end
    M=find(abs(Rmin-1)<=C);
    if (~isempty(M))
        Smax=max(Smin(M)); ihat=find(Smin(M)==Smax);
        %ghat = M(ihat(1));
        ii=randi(length(ihat));
        ghat=M(ihat(ii));
        
        ii=find(G~=ghat);N=G(ii);
        Sigmamin=min(Sigma(ghat,N)); imin=find(Sigma(ghat,N)==Sigmamin);
        %if length(M)>1 || t<15, %(Sigmamin<B),
        if (length(M)==1 && t>3*log2(length(G))) ||  Sigmamin>B
            nn=min(num_vstar_rated(ghat)+1,num_vstar_items); num_vstar_rated(ghat)=nn;
            l = vstar(ghat,nn);            
            while I(l)>0 &&  nn<num_vstar_items
                nn=min(num_vstar_rated(ghat)+1,num_vstar_items); num_vstar_rated(ghat)=nn;
                l = vstar(ghat,nn); 
            end
            IIhat=[IIhat;t,ghat,1,length(M)];
        else
            ii=randi(length(imin));
            nn=min(num_d_rated(ghat,N(imin(ii)))+1,num_items); num_d_rated(ghat,N(imin(ii)))=nn;
            l= d(ghat,N(imin(ii)),nn); 
            while I(l)>0 && nn<num_vstar_items
                nn=min(num_d_rated(ghat,N(imin(ii)))+1,num_items); num_d_rated(ghat,N(imin(ii)))=nn;
                l= d(ghat,N(imin(ii)),nn); 
            end
            IIhat=[IIhat;t,ghat,0,length(M)];
        end
    else
        ss=1e9; ghat=1; hhat=2;
        for g=G
            for h=G
                if (h==g), continue; end
                if Sigma(g,h)<ss
                    ghat=g; hhat=h; ss=Sigma(g,h);
                end
            end
        end
        nn=min(num_d_rated(ghat,hhat)+1,num_items); num_d_rated(ghat,hhat)=nn;
        l= d(ghat,hhat,nn);        
        while I(l)>0 && nn<num_vstar_items
            nn=min(num_d_rated(ghat,hhat)+1,num_items); num_d_rated(ghat,hhat)=nn;
            l= d(ghat,hhat,nn);     
        end
        IIhat=[IIhat;t,ghat,0,length(M)];
    end
    %pull arm
    X = Ruser(l);
    % update count of ratings of items
    I(l) = I(l)+1;
    % baseline
    nn2=min(num_vstar_rated2+1,num_vstar_items); num_vstar_rated2=nn2;
    Xstar = Ruser(vstar(gg,nn2));
    regret = regret + X-Xstar; %mu(gg,vstar(gg));
    regret2 = regret2 + mu(gg,l)-mu(gg,vstar(gg,nn2));
    LL=[LL;l,vstar(gg,nn2),X,Xstar,mu(gg,l),mu(gg,vstar(gg,nn2))];
    %history_reward=[history_reward;X];
    %history_item=[history_item;l];
    
    % update Sigma
    Sigma=Sigma+gammagh(:,:,l);
    ii=find(isinf(gammagh(:,:,l)));
    if (length(ii)>0)
        [inf,l,ghat],gammagh(:,:,l)
    end
    % update R
    for g=G
        for h=G
            if (h==g), continue; end
            Dg_sum(g,h)=Dg_sum(g,h)+gammagh(g,h,l)*(X-mu(h,l))/(mu(g,l)-mu(h,l));
        end
    end
    R=abs(Dg_sum)./Sigma;
    % update S
    S = R.*sqrt(2*Sigma);
    Smin = zeros(1,length(G));
    for g = G
        ii = find(G ~= g);
        Smin(g)=min(S(g,ii));
    end
    
    XX=[XX;t,regret,regret2,gg==ghat];
    %max(num_d_rated)
    if max(max(num_d_rated))==num_items, disp('newAlgo: hit max num_d_rated! increase num_items.'); end
end
for l=find(I>1)
    if I(l)>1 % sanity check: we should only have rated each item at most once
        fprintf("newAlgo: problem! %g pulled %g times\n",l,I(l))
    end
end
maxregret=max(XX(end,3));
t_converge = find(XX(:,3)/maxregret>=0.8, 1);
%t_converge = find(XX(:,4)/XX(end,4)>=0.8, 1);
if isempty(t_converge), t_converge=NaN; end
output=table(XX(:,2),XX(:,3),XX(:,4),t_converge*ones(length(XX),1),'VariableNames',{'regret','mean_regret','ghat_correct','t_converge'});

return

% plot some diagnostics
figure(1)
plot(XX(:,1),XX(:,2),XX(:,1),XX(:,3),'.',IIhat(:,1),IIhat(:,2),'o',IIhat(:,1),IIhat(:,3),'--',IIhat(:,1),IIhat(:,4),'o-','LineWidth',3)
xlabel('time'),ylabel('regret'),
set(gca,'fontsize',24)
figure(2)
plot(XX(:,1),XX(:,3+gg),'.',XX(:,1),XX(:,5),'o','LineWidth',3)
xlabel('time'),ylabel('R_t(g)'), legend('R_t(gg)','R_t(2)')
%ylim([0 4])
set(gca,'fontsize',24)

II=NaN(length(G),length(G));
for i=1:length(items), [g,h]=find(d==items(i)); for j=1:length(g),II(g(j),h(j))=I(i);end; end
II
II=NaN(1,length(G));
for i=1:length(items), [g,h]=find(vstar==items(i)); for j=1:length(g),II(g(j),h(j))=I(i);end; end
II