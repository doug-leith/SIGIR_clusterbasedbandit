% use a group of items for each distinguisher
function [dd2,vstar2,items,gammagh]= finddistinguishers2(mu,sigma,num,G,num_items,num_vstar_items,min_ratings)

    %num_items number of items in each distinguisher group/arm 
    %num_vstar_items number of items for each optimum arm
    gammagh2=NaN(length(G),length(G),num_items);
    dd2=NaN(length(G),length(G),num_items);
    ii=find(min(num)>=min_ratings);
    for g = G
        for h = G
            if (g==h), continue; end
            %ii1=find(min([num(g,:);num(h,:)])>=max(1,min_ratings)); % dating
            ii1=ii;
            % maxk returns top k items with k=num_items, sorted in descending order
            [gammagh2(g,h,:),items] = maxk( (mu(g,ii1)-mu(h,ii1)).^2./(8*sigma(g,ii1)), num_items);
            dd2(g,h,:)=ii1(items);
        end
    end
    %gammagh2

    %find optimum arms, must have gamma(g,h)>0 for all g,h to be admissible
    vstar2=NaN(length(G),num_vstar_items);
    %ii=find(min(num)>=5);
    for g = G
        %ii1=find(num(g,:)>=max(1,min_ratings)); % dating
        ii1=ii;
        [vals,s]=sort(mu(g,ii1),'ascend'); % sort items by descending mean rating
        kk=find(G~=g);N=G(kk);
        items=[];
        for v = s
            GG=(mu(g,ii1(v))-mu(N,ii1(v)));
            %if length(GG)<length(G)-1, continue; end % groups with same diff, no good
            if min(abs(GG)>0.01) 
                % add item to list associated with optimum arm for group g
                items=[items,ii1(v)]; 
                if length(items)==num_vstar_items, break; end % we have enough items, move on
            end
        end
        vstar2(g,:)=items;
    end
    %vstar2
    %[diag(mu(:,vstar2))]

    % get the set of distinguisher and optimal items ...
    ii=find(~isnan(dd2));items=unique(dd2(ii))';  % diaginals of dd2 are NaN
    items=unique([items,unique(vstar2)']);
    
    % calc gamma_{g,h}
    %gammagh=NaN(length(G),length(G),max(items));
    gammagh=zeros(length(G),length(G),max(items));
    for g = G,
        for h = G,
            if (g==h), continue; end
            for v = items,
                gammagh(g,h,v)=(mu(g,v)-mu(h,v))^2/(8*sigma(g,v));
            end
        end
    end