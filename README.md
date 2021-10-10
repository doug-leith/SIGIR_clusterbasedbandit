
# ``Cluster-Based Bandits: Fast Cold-Start for Recommender System New Users'', Proc SIGIR 2021


[Paper pdf](sigir16.pdf?raw=true) and [Video of SIGIR'21 presentation](https://github.com/doug-leith/SIGIR_clusterbasedbandit/blob/main/sigir2021.mp4?raw=true)

Code is implemented in matlab rather than python.  To run first clone the repository, then navigate to local folder containing the files and:

1. unzip netflix_full_8nyms.zip
2. open matlab
3. open file test_netflix_ng2.m in matlab, and run

The test_netflix_ng2.m will load in the data for the Netflix data with 8 clusters from the netflix_full_8nyms folder (that you obtained by unzipping netflix_full_8nyms.zip), train a decision tree on the data then run the decision tree and cluster bandit for a set of randomly generated users and output performance stats.

Since training the decision tree is time consuming its best to only do that once for each data set.  To disable training change "train_DT=1" to "train_DT=0" in test_netflix_ng2.m

Change the value of "Nyms=8" e.g. to "Nyms=16" will cause the the Netflix data with 16 clusters from the netflix_full_16nyms to be used (you'll need to unzip netflix_full_16nyms.zip first of course).  To use other datasets (Jester, GoodReads) just uncomment the relevant code in test_netflix_ng2.m and/or 

To plot extra results: for time histories of accuracy change "plot_accuracy=0" to "plot_accuracy=1", for convergence time data change "plot_t_converge=0" to "plot_t_converge=1"
