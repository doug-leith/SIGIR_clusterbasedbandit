function [tree,ii] = trainDecisionTree(G,mu,sigma,num,min_ratings)
    fprintf("DT: building R\n");
    ii=find(min(num)>=min_ratings);
    num_items=length(ii); num_groups=length(G);
    num_users=1000;
    R=NaN(num_users*num_groups,num_items); Group=NaN(num_users*num_groups,1);
    for g=G
        for u=1:num_users
            Group((g-1)*num_users+u) = g;
            R((g-1)*num_users+u,:) = sqrt(sigma(g,ii)).*randn(1,num_items)+mu(g,ii);
        end
    end
    fprintf("DT: fitting tree\n");
    %tree = fitctree(R,Group,'MaxNumSplits',num_groups^3);
    tree = fitctree(R,Group);
    %view(tree,'Mode','Graph')

    return
    
    % for testing
    fprintf("DT: generating test data\n");
    num_test_users=floor(num_users);
    Rtest=NaN(num_test_users*num_groups,num_items); Grouptest=NaN(num_test_users*num_groups,1);
    for g=G
        for u=1:num_test_users
            Grouptest((g-1)*num_test_users+u) = g;
            Rtest((g-1)*num_test_users+u,:) = sqrt(sigma2(g,ii)).*randn(1,num_items)+mu(g,ii);
        end
    end
    %confusionmat(Grouptest,Grouppreds)

    fprintf("DT: making predictions\n");
    Grouppreds=predict(tree,Rtest);
    confusionmat(Grouptest,Grouppreds)
