import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import math 
from datetime import datetime
from pytz import timezone
from IPython.display import display

from utils.df_pca import DfPca 
from utils.df_data_prep_01 import DfDataPrep01
from utils.df_data_prep_02 import DfDataPrep02

# -------------------------------------------------------------------
# constants  
# -------------------------------------------------------------------
GC_MAX_ROWS = 10000
GC_MAX_MISSING_RATE = 0.25
GC_UTC = timezone('UTC')
GC_BASE_COLOR = sns.color_palette()[0]
GC_COLORMAP = plt.cm.RdBu

class DfProj: 
    """ 
    class with some helper methods to print statistics around a dataset 
    """
    
    def __init__(self, p_print=True, p_plot=True, p_max_rows=GC_MAX_ROWS):
        """
        Instantiates the object with the dataset 
        Input: 
        - Dataset to analyze
        """
        self.set_print(p_print)
        self.set_plot(p_plot) 
        self.set_max_rows(p_max_rows)

    def set_print(self, p_print): 
        self.me_print = p_print 

    def set_plot(self, p_plot): 
        self.me_plot = p_plot 

    def set_max_rows(self, p_max_rows): 
        self.me_max_rows = p_max_rows 
# -------------------------------------------------------------------
# extract the master data from the md-file 
# -------------------------------------------------------------------
    def get_data( self ): 
        """
        load the general population and the customer data + the feature attribut descriptions
        """
        self.print_progress('FREE', 'Load Data') 
        # Load in the general demographics and the customer data.
        if self.me_max_rows > 0:
            self.azdias = pd.read_csv('./data/Udacity_AZDIAS_052018.csv', 
                        delimiter=';', nrows = self.me_max_rows)
            self.customers = pd.read_csv('./data/Udacity_CUSTOMERS_052018.csv', 
                        delimiter=';', nrows = self.me_max_rows)
        else:
            self.azdias = pd.read_csv('./data/Udacity_AZDIAS_052018.csv', delimiter=';').sample(frac = 0.20)
            self.customers = pd.read_csv('./data/Udacity_CUSTOMERS_052018.csv', delimiter=';').sample(frac = 0.20)

        # Load in the feature summary file.
        self.feat_info = pd.read_excel('./data/DIAS Information Levels - Attributes 2017_CHANGED.xlsx', 
                header = 1, usecols = [1,2,3,4,5]).fillna(method = 'ffill')
        self.feat_info = self.feat_info.set_index('Attribute')
        self.feat_info.rename(columns={'CAMEO_DEUINTL_2015':'CAMEO_INTL_2015'},inplace=True)

        # load in the feature attributes file 
        self.feat_attr = pd.read_excel('./data/DIAS Attributes - Values 2017.xlsx', 
                    header = 1, usecols = [1,2,3,4]).fillna(method = 'ffill')
        self.feat_attr = self.feat_attr.set_index('Attribute')
        self.feat_attr.rename(columns={'CAMEO_DEUINTL_2015':'CAMEO_INTL_2015'},
                                inplace=True)
        print(' general / customer: ', self.azdias.shape, self.customers.shape, '\n' )

        # column dictionary - gets already here the missing column values 
        self.cols_dic_gen = self.get_cols_dic( self.azdias, GC_MAX_MISSING_RATE )
        df_cols_dic_gen = pd.DataFrame.from_dict(self.cols_dic_gen, orient='index')
        
        # load the master data
        self.master_dic = self.get_master_data()

        # initialize models_dic
        self.models_dic = {'m_pca': '', 'm_impmost': '', 'm_impmean': '', 
                            'm_scaler': '', 'm_onehot': ''  } 
        if self.me_print == True:  
            print('Loaded ----------------------------------------\n', 
                    ' general / customer: ', self.azdias.shape, self.customers.shape, '\n',
                    ' feature-info / attr: ', self.feat_info.shape, self.feat_attr.shape, '\n',
                    ' master / columns: ', len(self.master_dic), len(self.cols_dic_gen)
                    )

        return self.azdias, self.customers, self.feat_info, self.feat_attr, \
                self.master_dic, self.cols_dic_gen, self.models_dic 

# -------------------------------------------------------------------
# extract the training data  
# -------------------------------------------------------------------
    def get_data_train( self ): 
        """
        Load the mailout training data set 
        """
        self.print_progress('FREE', 'Load Mailout Train Data') 
        if self.me_max_rows > 0:
            self.mailout_train = pd.read_csv('./data/Udacity_MAILOUT_052018_TRAIN.csv', 
                    nrows=self.me_max_rows , sep=';')
        else:
            self.mailout_train = pd.read_csv('./data/Udacity_MAILOUT_052018_TRAIN.csv', 
                    sep=';')
        return self.mailout_train 

# -------------------------------------------------------------------
# extract the training data  
# -------------------------------------------------------------------
    def get_data_test( self ): 
        """
        Load the mailout test data set 
        """
        self.print_progress('FREE', 'Load Mailout Test Data') 
        if self.me_max_rows > 0:
            self.mailout_test = pd.read_csv('./data/Udacity_MAILOUT_052018_TEST.csv', 
                    nrows=self.me_max_rows , sep=';')
        else:
            self.mailout_test = pd.read_csv('./data/Udacity_MAILOUT_052018_TEST.csv', 
                    sep=';')
        return self.mailout_test  

# -------------------------------------------------------------------
# extract the master data from the md-file 
# -------------------------------------------------------------------
    def get_master_data( self ): 
        """
        # extract the master data from the md-file 
        """
        attr_dic = {}
        skip = True  
        # attr = line.rsplit(' ')[2]
        for key, attr in self.feat_attr.iterrows(): 
            attr_list = [] 
            # attr_key = attr.index + '|' + str(attr['Value']).lstrip()
            if attr['Meaning'] == 'unknown':
                attr_list = list( str(attr['Value']).lstrip().split(',' ))
            else: 
                attr_str = str(attr['Value']).lstrip()
                attr_list.append(attr_str)
            if len(attr_list) > 0:
                for attr_list_item in attr_list: 
                    attr_key = key + '|' + attr_list_item 
                    attr_value = attr['Meaning']
                    attr_dic[attr_key] = attr_value
        return attr_dic

# -------------------------------------------------------------------
# get all the columns out of the feature list and extract some info
# about the missing values and the column data type
# -------------------------------------------------------------------
    def get_cols_dic( self, p_df, p_nan_rate=0.5 ): 
        """
        # get all the columns out of the feature list and extract some info
        # about the missing values and the column data type
        """
        cols_type = p_df.dtypes
        cols_dic = {}
        row_cnt = p_df.shape[0]
        feat_unknown = self.feat_attr.query('Meaning == "unknown"')
        feat_numeric = self.feat_attr.query('Meaning == "numeric value"')
        for col_name in p_df.columns:
            # init's
            col_skip = ''
            col_skip_reason = '' 
            col_desc = ''
            col_not_in_dic = ''
            col_unknown = '' 
            col_datatype = '' 
            col_unknown_rate = 0.0
            col_unknown_cnt = 0.0
            col_unknown_list = []
            col_unknown_skip = False 
            # calculation
            col_type = cols_type[col_name]
            col_int = True if col_type == 'int64' or col_type == 'float64' else False 
            col_nan  = p_df[col_name].isna().sum()
            col_nan_ratio = col_nan / row_cnt 
            col_unique_cnt = p_df[col_name].nunique()
            col_unique = p_df[col_name].unique()[:5]
            if col_name not in self.feat_info.index: 
                col_not_in_dic = 'X'
                if self.me_print == True: 
                    print('column not in feature dictionary', col_name)
            if col_name in feat_unknown.index: 
                col_unknown = feat_unknown.loc[col_name].Value
            col_unknown_cnt, col_unknown_rate, col_unknown_list \
              = self.get_cols_missing( p_df, col_name, col_unknown, col_int)
            if col_name in self.feat_info.index: 
                col_desc = self.feat_info.loc[col_name].Description
                col_datatype = self.feat_info.loc[col_name].DataType
            if col_name in feat_numeric.index: 
                col_datatype = 'numeric'
            if col_type == 'object':
                col_datatype = 'categorial'
            if col_unknown_rate > p_nan_rate:
                col_unknown_skip = True
                col_skip_reason = 'high rate of NaNs'
                col_skip = True 
            cols_dic[col_name] = {'col_name': col_name, 'type': col_type , 'datatype': col_datatype, 
                                  'inttype': col_int, 'skip': col_skip, 'desc': col_desc, 
                                'nan_cnt': col_nan, 'nan_ratio': col_nan_ratio,
                                'unique_cnt': col_unique_cnt, 'unique': col_unique,
                                'unknown': col_unknown, 'unknown_cnt': col_unknown_cnt,
                                'unknown_rate': col_unknown_rate, 
                                'unknown_skip': col_unknown_skip,
                                'unknown_list': col_unknown_list, 
                                'skip_reason': col_skip_reason, 
                                'not_in_dic': col_not_in_dic } 
        return cols_dic

# -------------------------------------------------------------------
# convert column dictionary for certain columns into a dataframe
# -------------------------------------------------------------------
    def get_cols_dic_as_df(p_dic, p_cols):
        """
        convert into column dictionary into dataframe
        input: p_dic dictionary
            p_cols columns to export 
        export: column dictionary as dataframe  
        """
        if len(p_cols) == 0: 
            cols = ['type', 'skip', 'not_in_dic', 'nan_cnt', 'nan_ratio', 'unique_cnt', 'unknown_cnt', 'unknown_rate', 'unique']
        else:
            cols = p_cols
        df_cols_dic = pd.DataFrame.from_dict(p_dic, orient='index')
        return df_cols_dic[cols]

# -------------------------------------------------------------------
# get all the columns with missing values  
# -------------------------------------------------------------------
    def get_cols_missing( self, p_df, p_col_name, p_col_unknown, p_col_int ): 
        """
        # get all the columns with missing values dependent on the 
        # dictionary info about nan values 
        """
        missing_list = [] 
        missing_int = []
        missing_cnt = 0
        missing_rate = 0.0
        if len(str(p_col_unknown)) > 0: 
            # convert string to list 
            # missing_list = list(p_col_unknown.values[0].replace('[', '').replace(']', '').split(','))
            # convert search string to integer if necessary
            missing_list = list(str(p_col_unknown).split(','))
            if p_col_int == True:  
                for item in missing_list:
                    if item == 'X' or item == 'XX': item = '-1'
                    if item == 'XX': item = '-1'
                    # print('col', col_name, col_int )
                    if item != '':
                        missing_int.append(int(item))
                    missing_list = missing_int 
        missing_list.append(np.nan)
        # info about the missed values 
        missing_cnt = len(p_df.loc[p_df[p_col_name].isin(missing_list)]) 
        missing_rate =  missing_cnt / p_df.shape[0] 
        return missing_cnt, missing_rate, missing_list

# -------------------------------------------------------------------
# Perform feature trimming, re-encoding, and engineering for demographics
# data
# -------------------------------------------------------------------
    def clean_data(self, p_df, p_cols_dic, p_max_nan_columns):
        """
        Perform feature trimming, re-encoding, and engineering for demographics
        data
        
        INPUT: Demographics DataFrame
        OUTPUT: Trimmed and cleaned demographics DataFrame
        """
        
        # Put in code here to execute all main cleaning steps:
        # convert missing value codes into NaNs, ..
        cols_dic = {} 
        #  get infos about the column features
        if len(p_cols_dic) > 0:
            cols_dic = p_cols_dic  
            set_skip = False # do not change skip value for customer dataset
        else:
            cols_dic = self.get_cols_dic( p_df, GC_MAX_MISSING_RATE ) 
            set_skip = True  

        o_data_prep01 = DfDataPrep01(p_df, cols_dic, self.master_dic, self.models_dic, 
                                            p_print=self.me_print, p_plot=self.me_plot )
        df_prep = o_data_prep01.do_col_nans(GC_MAX_MISSING_RATE, set_skip) 

        df_prep, max_nan_columns = o_data_prep01.do_row_nans(df_prep, p_max_nan_columns) 

        o_data_prep02 = DfDataPrep02(p_df, cols_dic, self.models_dic,
                                            p_print=self.me_print, p_plot=self.me_plot )
        df_prep, models_dic_prep = o_data_prep02.do_reencode( df_prep) 

        # Return the cleaned dataframe.
        return df_prep, cols_dic, models_dic_prep, max_nan_columns

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
        elif p_me == 'PRE_CLEAN':
            print(tmst, ' Preprocessing - Cleaning ----------------------')
        elif p_me == 'PRE_REMROW': 
            print(tmst, ' Preprocessing - Remove Nan Rows ---------------')
        elif p_me == 'PRE_REENCODE':
            print(tmst, ' feature reencoding-----------------------------')
        elif p_me == 'PRE_IMPUTE':
            print(tmst, ' feature imputing ------------------------------')
        # elif p_me == 'GEN_PCA': 
        #     print(tmst, ' General PCA -----------------------------------')
        # elif p_me == 'GEN_KMEANS': 
        #     print(tmst, ' General KMeans---------------------------------')
        elif p_me == 'CUS_LOAD': 
            print(tmst, ' Customer Load Data-----------------------------')
        elif p_me == 'CUS_CLEAN':
            print(tmst, ' Customer Clean Data ---------------------------')
        # elif p_me == 'CUS_PCA': 
        #     print(tmst, ' Customer PCA ----------------------------------')
        # elif p_me == 'CUS_KMEANS': 
        #     print(tmst, ' Customer KMeans -------------------------------')
        elif p_me == 'TRAIN_CLEAN':
            print(tmst, ' Mailout Train Clean Data ----------------------')
        # elif p_me == 'TRAIN_PCA': 
        #     print(tmst, ' Mailout Train PCA -----------------------------')
        # elif p_me == 'TRAIN_KMEANS': 
        #     print(tmst, ' Mailout Train KMeans --------------------------')
        elif p_me == 'TRAIN_FIT': 
            print(tmst, ' Mailout Train Fit -----------------------------')
        elif p_me == 'TRAIN_PREDICT': 
            print(tmst, ' Mailout Train Predict -------------------------')
        elif p_me == 'READY': 
            print(tmst, ' Ready -----------------------------------------')
        elif p_me == 'FREE':
            line = ' -----------------------------------------------'
            rest_len = len(line) - len(p_title) 
            header = p_title + line[:rest_len]
            print(tmst, header) 

    def do_response(self, p_df):
        """
        Extract the reponse column into an own data set MAILOUT_RESPONSE_TRAIN and drop
        the column from the main data set 
        """
        # delete the response column but keep the values in an own dataframe  
        col_name = 'RESPONSE'
        if col_name in p_df.columns:
            self.mailout_response_train = pd.DataFrame(p_df[[col_name, 'LNR']])
        #   drop the response column for further processing
            p_df.drop([col_name], axis = 1, inplace=True)

        response_cnt = self.mailout_response_train[col_name][self.mailout_response_train[col_name] == 1].count()
        response_rate = response_cnt / self.mailout_response_train.shape[0]
        if self.me_print == True: 
            print('response cnt', response_cnt)
            print('response rate',response_rate)
            print('response values\n', self.mailout_response_train[col_name].value_counts())

            display(self.mailout_response_train.head() )
    
    # def do_clean_pca_kmeans(self, p_df, p_selected_pca_no, p_selected_cluster_no): 
    #     #  clean data 
    #     df_clean, self.cols_dic_gen, self.models_dic, GC_MAX_NAN_COLUMNS = \
    #         self.clean_data(p_df, self.cols_dic_gen, GC_MAX_NAN_COLUMNS)
    #     # impute data 
    #     df_imputed,  self.models_dic = DfDataPrep01.do_imputing_mean(
    #         df_clean, 'mean',self.models_dic ) 
    #     # Apply PCA to the data.
    #     pca, df_pca, scaler, self.models_dic = \
    #         DfPca.do_pca(p_selected_pca_no, df_imputed, self.models_dic, self.me_print )
    #     k_train_score, k_train_cluster, k_train_center, k_train_label = \
    #             o_df_kmeans.get_kmeans(df_pca, p_selected_cluster_no )
    #     return df_clean, df_imputed, df_pca 

    def do_cluster(self, p_df, p_pca, p_kmean_label, p_df_lnr, p_print): 
        """
        Collect the PCA components, the KMeans cluster no and the LNR reference into an own data frame
        delete some rows if the LNR is NaN 
        """
        # intersection of both dataframes - just in case rows werde deleted
        # during cleaning - the size of p_pca and p_df should be the same  
        # p_df = pd.merge(p_df, p_df_lnr['LNR'], how='inner', on=['LNR'] )
        p_df_lnr = pd.merge(p_df_lnr, p_df['LNR'].to_frame(), how='inner', on=['LNR'] )

        # df_filter_lnr = p_df_lnr['LNR'].isin(p_df['LNR'])
        df_cluster=p_pca 
        df_cluster.columns = ['PCA_' + str(x) for x in p_pca.columns]
        df_cluster['cluster_no'] = p_kmean_label
        df_cluster['RESPONSE'] = p_df_lnr['RESPONSE']
        df_cluster['LNR'] = p_df_lnr['LNR']
        # df_cluster_train['RESPONSE'] = p_df_lnr['RESPONSE'][df_filter_lnr]
        # df_cluster_train['LNR'] = p_df_lnr['LNR'][df_filter_lnr]

        col_name = 'RESPONSE'
        missing_cnt = len(df_cluster.loc[df_cluster[col_name].isna()])
        missing_rate = missing_cnt / p_df_lnr.shape[0]
        if self.me_print == True: 
            print('unique\n', df_cluster[col_name].value_counts(dropna=False))
            print('nan cnt', missing_cnt )
            print('nan rate',  missing_rate )
        # drop rows where LNR or response is nan 
        #if missing_cnt > 0:
        #    display(df_cluster.head())
        df_cluster = self.delete_nans( df_cluster, ['LNR'] )
        if self.me_print == True: 
            print('Mailout Cluster Shape', df_cluster.shape)
        return df_cluster, p_df_lnr 

    def print_cluster(self, p_df): 
        """
        Print some info about the cluster data frame
        """
        df_sum = p_df.groupby(['cluster_no', 'RESPONSE'])['RESPONSE'].count().to_frame() 
        df_sum.columns = ['cnt']
        print('sum', df_sum.shape)
        df_sum2 = df_sum.unstack(level=1) 
        # display(df_sum2.head())
        df_sum2['sum']  = df_sum2.iloc[:,0] + df_sum2.iloc[:,1] 
        df_sum2['ratio'] = df_sum2.iloc[:,1] / df_sum2.iloc[:,2] 
        # df_table2.reset_index(inplace=True)
        display(df_sum2.head())

        df_sum2.plot(kind='pie', x='cluster_no', y='sum', autopct='%1.0f%%',
                                colors = ['red', 'pink', 'steelblue'],
                                title='Responses by Cluster')
    
    def delete_nans( self, p_df, p_cols):
        """
        Drop NaNs for certain columns from the data set 
        """
        # check nan values 
        for col in p_cols:
            cnt_found = 0
            if col in p_df.columns:
                cnt_nan = p_df[col].isna().sum().sum()
                if cnt_nan > 0:
                    cnt_found += 1
                    if self.me_print == True: 
                        print('column with nans', col, ' No', cnt_nan)
            if cnt_found > 0:
                p_df.dropna(subset=[col], inplace=True)
        return p_df
