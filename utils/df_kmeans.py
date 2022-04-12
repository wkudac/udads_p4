# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import sklearn.cluster as cluster
import sklearn.metrics as metrics

from datetime import datetime
from pytz import timezone

# -------------------------------------------------------------------
# constants  
# -------------------------------------------------------------------
# GC_MAX_ROWS = 1000
# GC_MAX_MISSING_RATE = 0.25
GC_UTC = timezone('UTC')
GC_BASE_COLOR = sns.color_palette()[0]
GC_COLORMAP = plt.cm.RdBu

class DfKmeans: 
    """ 
    class with some helper methods to print statistics around a dataset 
    """
    
    def __init__(self, p_print=True, p_plot=True ):
        """
        Instantiates the object with the dataset 
        Input: 
        - Dataset to analyze
        """
        self.set_print(p_print)
        self.set_plot(p_plot) 

    def set_print(self, p_print): 
        self.me_print = p_print 

    def set_plot(self, p_plot): 
        self.me_plot = p_plot 

# -------------------------------------------------------------------
    def print_progress(self, p_me, p_title=''):
        """
        print the progress in a formatted way with a timestamp
        Input:
        p_me: current status
        p_title: printed if P_ME=FREE is used 
        """
        tmst =  datetime.now(GC_UTC)
        if p_me == 'PRE_REMCOL':
            print(tmst, ' Preprocessing - Remove Nan Colums -------------')
        elif p_me == 'GEN_KMEANS': 
            print(tmst, ' General KMeans---------------------------------')
        elif p_me == 'CUS_KMEANS': 
            print(tmst, ' Customer KMeans -------------------------------')
        elif p_me == 'FREE':
            line = ' -----------------------------------------------'
            rest_len = len(line) - len(p_title) 
            header = p_title + line[:rest_len]
            print(tmst, header) 

# -------------------------------------------------------------------
# get the kmeans score values for a target clsuter (n_comp)
# -------------------------------------------------------------------
    def get_kmeans(self, p_df, p_cluster_no):
        """
        # get the kmeans score values for a target clsuter (n_comp)
        """
        title = 'KMeans ' + str(p_cluster_no) 
        self.print_progress('FREE', title) 

        k_score = 0
        k_cluster = []
    #     kmeans = KMeans(cluster_no)
    #     k_model = kmeans.fit(p_df)
        batch_size = 1024 # 100
        kmeans = MiniBatchKMeans(n_clusters=p_cluster_no, random_state=0, batch_size=batch_size)     
        k_model = kmeans.fit(p_df)
        k_label = k_model.predict(p_df)
        # print('label', len(k_label), k_label)
        # print('df', p_df.shape)
        k_center = k_model.cluster_centers_
        unique, counts = np.unique(k_label, return_counts = True )
        k_cluster = np.column_stack((unique, counts))
        k_score = np.abs(k_model.score(p_df))
        if self.me_print == True:
            print('KMeans - Shape ', kmeans.cluster_centers_.shape, k_model.inertia_, 'No of Labels: ', 
                    len(k_label), ' Score: ', k_score, ' DataFrame: ', p_df.shape)
        # print('KMeans - Center', k_model.cluster_centers_)
        return k_score, k_cluster, k_center, k_label 

# -------------------------------------------------------------------
# get the original feature weights of a certain kmeans cluster 
# back from the center values 
# -------------------------------------------------------------------
    def get_kmeans_inverse(self, p_model_pca, p_scaler, p_center, p_df):
        """
        # get the original feature weights of a certain kmeans cluster 
        # back from the center values 
        """
        pca_inverse = p_model_pca.inverse_transform(p_center)
        pca_scaled = p_scaler.inverse_transform(pca_inverse)
        overrep_df = pd.DataFrame(pca_scaled, index=p_df.columns, columns = ['Weight'])
        df_feat = overrep_df.sort_values(by = 'Weight', ascending = False)
        return df_feat

# -------------------------------------------------------------------
# get the kmeans score values for a number of chosen target
# clusters (n_comp)  
# -------------------------------------------------------------------
    def get_kmeans_scores(self, p_df, p_cluster_nos):
        """
        # get the kmeans score values for a number of chosen target
        # clusters (n_comp)  
        """
        k_scores = []
        k_clusters = []
        k_centers = []
        k_labels = []
        cluster_no_max = 30
        for ind in p_cluster_nos: # range(1,cluster_no):
            k_score, k_cluster, k_center, k_label = self.get_kmeans(p_df, ind) 
            # no of distinct clusters found smaller than target cluster no -> stop  
            if len(k_cluster) > cluster_no_max: 
                break
            k_scores.append(k_score) 
            k_clusters.append(k_cluster) 
            k_centers.append(k_center)
            k_labels.append(k_label) 
            # print('Cluster:' , k_cluster.reshape(1,-1))
            # print('Score', ind, k_score)
        return k_scores, k_clusters, k_centers, k_labels

# silhouette score to get the best cluster no 
# The silhouette value measures the similarity of a data point within its cluster. 
# It has a range between +1 and -1 and the higher values denote a good clustering.
# -------------------------------------------------------------------
    def get_kmeans_silhouette(self, p_df_pca, p_cluster_nos): 
      for i in range(2,p_cluster_nos):
          labels=cluster.KMeans(n_clusters=i,random_state=200).fit(p_df_pca).labels_
          if self.me_print == True:
            print ("Silhouette score for k(clusters) = "+str(i)+" is "
                    +str(metrics.silhouette_score(p_df_pca,labels,metric="euclidean",
                    sample_size=1000,random_state=200)))

# -------------------------------------------------------------------
# plot the kmeans scores - elbough plot   
# -------------------------------------------------------------------
    def plot_kmeans_cluster(self, p_cluster, p_cluster2, p_col_label1, p_col_label2):
        """
        # plot the kmeans scores - elbough plot   
        """
        # ind = p_cluster[:, 0] 
        ind = np.arange(len(p_cluster[:, 1]))
        vals = p_cluster[:, 1]  
        num_x = len(ind)
        # ind = np.arange(num_comp)
        # vals = pca.explained_variance_ratio_
        width = 0.25 

        plt.figure(figsize=(16,10 ))
        ax = plt.subplot(111)
        # cumvals = np.cumsum(vals)
        ax.bar(ind,vals, color = 'b', width=width, edgecolor = 'black', label=p_col_label1)
        if len(p_cluster2) > 0:
            vals2 = p_cluster2[:, 1]  
            ax.bar(ind + width + 0.05,vals2, color = 'g', width = width, edgecolor = 'black', 
                  label=p_col_label2)
        ax.plot(ind)
        for ii in range(num_x):
            val = "{:+.2f}".format(vals[ii])
            ax.annotate(val, xy=(ind[ii], vals[ii]), va="bottom", ha="center", fontsize=11)
            if len(p_cluster2) > 0: 
                val2 = "{:+.2f}".format(vals2[ii])
                ax.annotate(val2, xy=(ind[ii]+width + 0.05, vals2[ii]), va="bottom", ha="center", 
                fontsize=11)

        # ax.xaxis.set_tick_params(width=0)
        # ax.yaxis.set_tick_params(width=1, length=12)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(p_col_label2)
        plt.xticks(ind + width/2,ind) 
        plt.title('Elements per Cluster') 
        plt.legend()
        plt.show()

# -------------------------------------------------------------------
# plot the kmeans scores - elbough plot   
# -------------------------------------------------------------------
    def plot_kmeans_scores(self, p_cluster_nos, p_scores):
        """
        # plot the kmeans scores - elbough plot   
        """
        centers = p_cluster_nos # list(range(1,n_comp))

        plt.figure(figsize=(10,6 ))
        plt.plot(centers, p_scores)
        plt.title('KMeans')
        plt.xlabel('Centers')
        plt.ylabel('Average Distance from Centroid')
        plt.show() 

    def plot_kmeans_centers(self, p_centers, p_cols):
        if self.me_print == True:
            print('centers', p_centers.shape, p_cols)
        centers = pd.DataFrame(p_centers, columns=p_cols)
        
        f, axes = plt.subplots(20,1,figsize=(40,80), sharex=True) 
        for i, ax in enumerate(axes):
            center = centers.loc[i, :]
            maxPC = 1.01 * np.max(np.max(np.abs(center))) 
            colors = ['C0' if l > 0 else 'C1' for l in center]
            ax.axhline(color='#888888')
            center.plot.bar(ax=ax, color=colors)
            ax.set_ylabel(f'Cluster {i + 1}') 
            ax.set_ylim(-maxPC, maxPC) 