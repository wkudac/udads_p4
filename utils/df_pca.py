import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import math 
from datetime import datetime
from pytz import timezone
from IPython.display import display

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 

# -------------------------------------------------------------------
# constants  
# -------------------------------------------------------------------
# GC_MAX_ROWS = 1000
# GC_MAX_MISSING_RATE = 0.25
GC_UTC = timezone('UTC')
GC_BASE_COLOR = sns.color_palette()[0]
GC_COLORMAP = plt.cm.RdBu

class DfPca: 
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

# -------------------------------------------------------------------
    def print_progress(p_me, p_title=''):
        """
        method to print the progress in a formatted way with timestamp and step info
        Input:
        - p_me: which step is currently done
        - p_title: will be printed for the FREE option 
        """
        tmst =  datetime.now(GC_UTC)
        if p_me == 'PRE_REMCOL':
            print(tmst, ' Preprocessing - Remove Nan Colums -------------')
        elif p_me == 'CUS_PCA': 
            print(tmst, ' Customer PCA ----------------------------------')
        elif p_me == 'GEN_PCA': 
            print(tmst, ' General PCA -----------------------------------')
        elif p_me == 'FREE':
            line = ' -----------------------------------------------'
            rest_len = len(line) - len(p_title) 
            header = p_title + line[:rest_len]
            print(tmst, header) 

# -------------------------------------------------------------------
# do pca transformation - feature reduction  
# -------------------------------------------------------------------
    @classmethod
    def do_pca(self, n_comp, p_df, p_models_dic, p_print): 
        """
        # do pca transformation - feature reduction  
        """
        self.print_progress('GEN_PCA')
        scaler_obj = StandardScaler()
        # if p_scaler_model == '':
        if p_models_dic['m_scaler'] == '': 
            if p_print == True:
                print('new scaler')
            scaler_model = scaler_obj.fit(p_df) 
            p_models_dic['m_scaler'] = scaler_model # p_models_dic['m_scaler'] # scaler_model
        else:
            if p_print == True:
                print('old scaler')
            scaler_model = p_models_dic['m_scaler'] # p_scaler_model 
        X_scaled = scaler_model.transform(p_df)

        pca_obj = PCA(n_comp)
        # if p_pca_model == '':
        if p_models_dic['m_pca'] == '': 
            if p_print == True:
                print('new pca model')
            pca_model = pca_obj.fit(X_scaled)
            p_models_dic['m_pca'] = pca_model
        else:
            if p_print == True:
                print('old pca model')
            pca_model = p_models_dic['m_pca'] # p_pca_model  
        X_pca = pca_model.transform(X_scaled) 
        
        # pca_model = pca.fit_transform(X_scaled)
        return pca_obj, pd.DataFrame(X_pca), scaler_obj, p_models_dic

  # -------------------------------------------------------------------
  # do pca transformation with an existing model - feature reduction  
  # -------------------------------------------------------------------
    @classmethod
    def do_pca_model( self, p_df, p_model): 
        """
        # do pca transformation with an existing model - feature reduction  
        """
        X_scaled = StandardScaler().fit_transform(p_df)
        X_pca =p_model.transform(X_scaled) 
        # pca_model = pca.fit_transform(X_scaled)
        return pd.DataFrame(X_pca), pd.DataFrame(X_scaled) 

# Reporting
# Investigate the variance accounted for by each principal component.

# hear only the top feature is displayed together with the weight. 
# later more features of one PC will be displayed too (s. below)
# -------------------------------------------------------------------
# print the pca values for all principal components
# -------------------------------------------------------------------
    @classmethod
    def print_pca(p_model_pca, p_comp_no, p_df):
        """
        # print the pca values for all principal components
        """
        feat_names_init = p_df.columns 

        # # most important feature for every pc
        most_important = [p_model_pca.components_[ii].argmax() for ii in range(p_comp_no)]

        # get the names
        feat_names_important = [feat_names_init[most_important[ii]] for ii in range(p_comp_no)]
        feat_values_important = [np.abs(p_model_pca.components_[ii]).max() for ii in range(p_comp_no)]
        feat_important = list(zip(feat_names_important, feat_values_important)) 
        
        dic2 = {'PC{}'.format(ii+1): feat_important[ii]  for ii in range(p_comp_no)}
        # build the dataframe
        # df = pd.DataFrame(list(dic2.items()),columns=['PCA', 'Feature, Weight'] ) 
        # print('Most important feature per Component by Weights\n', df.head( p_comp_no)) 
        
        feat_important2 = zip(feat_names_important, feat_values_important)
        df_pca = pd.DataFrame(feat_important2, columns=['Feature', 'Max Weight'] ) 
        df_pca.index.names = ['PCA']
        print('Most important feature per Component by Weights\n', p_model_pca.head( p_comp_no)) 
        # df2 = pd.DataFrame(dic2.items(),columns=['PCA', 'Feature, Weight'] ) 
        # print(df2)

    @classmethod
    def plot_pca( cls, p_pca, p_df, p_cols_dic): 
        """
        Plot the PCA components - 10 top/bottom weigths 
        """
        df_cumvals = cls.plot_pca_variance(p_pca)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        # print(df_cumvals.head(20))

    # Reporting 
    # 10 top and 10 bottom weigths for the PCA's  
        max_pca_plot = 10  
        for ii in range(max_pca_plot):
            cls.plot_pca_component(p_pca, ii, p_df )
            print('Dataset Shape: ', p_df.shape, ' No of Components: ', 
                len(p_pca.components_), ' No of Weights: ', len(p_pca.components_[ii]))

        df_cols_dic_gen = pd.DataFrame.from_dict(p_cols_dic, orient='index')
        for ii in range(5):
            display(cls.print_pca_one2(p_pca, ii, p_df, df_cols_dic_gen,5))

# -------------------------------------------------------------------
# plot the elbow curve for the PCA model  
# --> "scree_plot" function of the course 
# -------------------------------------------------------------------
    @classmethod
    def plot_pca_variance(self, p_model_pca):
        """
        # plot the elbow curve for the PCA model  
        # --> "scree_plot" function of the course 
        """
        num_comp = len(p_model_pca.explained_variance_ratio_ )
        ind = np.arange(num_comp)
        vals = p_model_pca.explained_variance_ratio_
        cumvals = np.cumsum(vals)
        df_cumvals = pd.concat([pd.DataFrame(ind), pd.DataFrame(vals), pd.DataFrame(cumvals)], axis = 1) 
        df_cumvals.columns = [ 'Component', 'Variance', 'Cumul.Variance']
        
        plt.figure(figsize=(16,8 ))
        ax = plt.subplot(111)
      
        ax.bar(ind,vals)
        ax.plot(ind,cumvals)
        for i in range(num_comp):
            ax.annotate(r"%s%%" % ((str(vals[i]*100))[:4]), (ind[i]+0.2, vals[i]), 
                        va="bottom", ha="center", fontsize=8, rotation=60)
        ax.xaxis.set_tick_params(width=0)
        ax.yaxis.set_tick_params(width=2, length=12)

        ax.set_xlabel("Pricipal Component")
        ax.set_ylabel("Variance Explainded (%)")
        plt.title('Explained Variance Per Principal Component') 
        plt.show()
        return df_cumvals 
        # (rotation=30, ha='right', fontsize=8)

# -------------------------------------------------------------------
# plot the PCA components 
# -------------------------------------------------------------------
    @classmethod
    def plot_pca_component(self, p_model_pca, p_comp_no, p_df):
        """
        # plot the PCA components 
        """
        # df_plot = pd.DataFrame()
        # df_plot_sorted = pd.DataFrame()
        # df_res = pd.DataFrame()
        df_plot = pd.DataFrame(list(zip(p_df.columns, p_model_pca.components_[p_comp_no])), 
                              columns=['col_name', 'weight' ])
        df_plot_sorted = df_plot.sort_values(['weight'], ascending = False)
        df_res = pd.concat([df_plot_sorted[:10], df_plot_sorted[-10:]]) 
        ii = 0
    
        for item in df_res['col_name']: 
            df_res['col_name'].iloc[ii] = str(ii).rjust(2,'0') + '_' + item
            ii += 1
        vals = list(df_res['weight'])
        ind = list(df_res['col_name'])
        plt.figure(figsize=(16,8 ))
        plt.style.use('seaborn')
        ax = plt.subplot(111)
        for ind_res in range(len(df_res)):
            val = "{:+.2f}".format(vals[ind_res])
            ax.annotate(val, xy=(ind[ind_res], vals[ind_res]), va="bottom", ha="center", fontsize=10)
        cumvals = np.cumsum(vals)
        ax.bar(ind,vals)
        # ax.plot(ind,cumvals)
        ax.set_xticks(ind)
        ax.set_xlabel("Features")
        ax.set_ylabel("Weights")
        # plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.2)
        plt.xticks(rotation=50, ha='right', fontsize=11)
        plt.title('Top {} Features, Bottom {} Features for Component {}'.format(10, 10, p_comp_no)) 
        plt.show()

# -------------------------------------------------------------------
# print the pca weights for 1 chosen principal component sorted 
# by the weights value 
# -------------------------------------------------------------------
    @classmethod
    def print_pca_one(self, p_model_pca, p_comp_no, p_df): 
        """
        # print the pca weights for 1 chosen principal component sorted 
        # by the weights value 
        """
        feat_names_init = p_df.columns 
        feat_values_init = p_model_pca.components_[p_comp_no]
        feat = list(zip(feat_names_init, feat_values_init))
        feat_size = len(feat) 
        feat.sort(key = lambda i: i[1], reverse=True)
        print('Principal Component {} by Weigths\n'.format(p_comp_no))
        [print(feat[i][0], "%0.5f" % feat[i][1]) for i in range(feat_size)]
        df_feat = pd.DataFrame(feat, columns=['Feature', 'Weight']) 
        print(df_feat.head())

    @classmethod
    def print_pca_one2(self, p_model_pca, p_comp_no, p_df, p_df_cols_dic_gen, n_rows=10):
        """
        print the pca components sorted 
        """
        print('Principal Component {} by Weigths - Top/Bottom 10 \n'.format(p_comp_no))
        df_print = pd.DataFrame(list(zip(p_df.columns, p_model_pca.components_[p_comp_no])), 
                                columns=['col_name', 'weight' ])
        # sort by weights descending
        df_print_sorted = df_print.sort_values(['weight'], ascending = False)
        # extract top/bottom top by weights
        df_res = pd.concat([df_print_sorted[:n_rows], df_print_sorted[-n_rows:]]) 
        # cut last 2 columns for imputed columns - LP_STATUS_FEIN_1 --> LP_STATUS_FEIN 
        df_res['col_name_alt'] = df_res['col_name'].map(lambda xx: xx[:-2])
        # extract certain columns 
        df_print_dic = p_df_cols_dic_gen[['col_name', 'desc', 'unknown_rate', 'unique']]
        # merge df and column dic on column name
        df_res_1 = df_res.merge(df_print_dic, how='left', on='col_name')
        # rename columns
        df_print_dic.columns = ['col_name_alt', 'desc_alt', 'unknown_rate_alt', 'unique_alt']
        # merge df and column dic on reduced column name (imputed)
        df_res_2 = df_res.merge(df_print_dic, how='left', on='col_name_alt')
        cols = list(df_print_dic.columns.values)
        # concatenate both merges 
        df_res = pd.concat([df_res_1, df_res_2[cols]], axis=1) 
        # if column value is Nan - take the alternative values 
        df_res['desc'] = np.where(pd.isna(df_res.desc),df_res['desc_alt'], df_res['desc'])
        df_res['unknown_rate'] = np.where(pd.isna(df_res.unknown_rate),df_res['unknown_rate_alt'], 
                                df_res['unknown_rate'])
        df_res['unique'] = np.where(pd.isna(df_res.unique),df_res['unique_alt'], df_res['unique'])
        # return only certain columns
        cols = ['col_name', 'weight', 'desc', 'unknown_rate', 'unique' ]
        return df_res[cols]