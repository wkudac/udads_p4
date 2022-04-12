import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import math 
from datetime import datetime
from pytz import timezone
from IPython.display import display

from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder

# -------------------------------------------------------------------
# constants  
# -------------------------------------------------------------------
GC_MAX_ROWS = 1000
GC_MAX_MISSING_RATE = 0.25
GC_UTC = timezone('UTC')
GC_BASE_COLOR = sns.color_palette()[0]
GC_COLORMAP = plt.cm.RdBu

class DfDataPrep01: 
    """ 
    class to do some general specific data preparation 
    """
    
    def __init__(self, p_df, p_cols_dic, p_master_dic, p_models_dic,  p_print=True, p_plot=True ):
        """
        Instantiates the object with the dataset 
        Input: 
        - Dataset to analyze
        """
        self.df = p_df
        self.cols_dic = p_cols_dic 
        self.master_dic = p_master_dic 
        self.models_dic = p_models_dic 
        self.set_print(p_print)
        self.set_plot(p_plot) 

    def set_print(self, p_print): 
        self.me_print = p_print 

    def set_plot(self, p_plot): 
        self.me_plot = p_plot 

# -------------------------------------------------------------------
    @classmethod
    def print_progress_c(cls, p_me, p_title=''): 
        """
        static method to print the progress in a formatted way with timestamp and step info
        Input:
        - p_me: which step is currently done
        - p_title: will be printed for the FREE option 
        """
        tmst =  datetime.now(GC_UTC)
        if p_me == 'PRE_IMPUTE':
            print(tmst, ' feature imputing ------------------------------')
        elif p_me == 'READY': 
            print(tmst, ' Ready -----------------------------------------')
        elif p_me == 'FREE':
            line = ' -----------------------------------------------'
            rest_len = len(line) - len(p_title) 
            header = p_title + line[:rest_len]
            print(tmst, header) 

    def print_progress(self, p_me, p_title=''):
        """
        object method to print the progress in a formatted way with timestamp and step info
        Input:
        - p_me: which step is currently done
        - p_title: will be printed for the FREE option 
        """
        tmst =  datetime.now(GC_UTC)
        if p_me == 'PRE_REMCOL':
            print(tmst, ' Preprocessing - Remove Nan Colums -------------')
        elif p_me == 'PRE_CLEAN':
            print(tmst, ' Preprocessing - Cleaning ----------------------')
        elif p_me == 'PRE_REMROW': 
            print(tmst, ' Preprocessing - Remove Nan Rows ---------------')
        elif p_me == 'PRE_REENCODE':
            print(tmst, ' Feature Reencoding-----------------------------')
        elif p_me == 'PRE_IMPUTE':
            print(tmst, ' Feature Imputing ------------------------------')
        elif p_me == 'CUS_LOAD': 
            print(tmst, ' Customer Load Data-----------------------------')
        elif p_me == 'CUS_CLEAN':
            print(tmst, ' Customer Clean Data ---------------------------')
        elif p_me == 'CUS_PCA': 
            print(tmst, ' Customer PCA ----------------------------------')
        elif p_me == 'CUS_KMEANS': 
            print(tmst, ' Customer KMeans -------------------------------')
        elif p_me == 'TRAIN_CLEAN':
            print(tmst, ' Mailout Train Clean Data ----------------------')
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

# ----------------------------------------------------------------
# Execution - data cleaning - nan handling for columns 
# Identify missing or unknown data values and convert them to NaNs.
# ----------------------------------------------------------------
    def do_col_nans( self, p_missing_rate, p_set_skip ):
        """
        handle all the Nan handling for the columns 
        Input: 
        p_missing_rate: defines the rate when the columsn should be deleted 
                        because of a high no of missing values
        p_set_skip: True=change the skip flag if necessary, False=Do not change the skip flag
        """
        self.print_progress('PRE_REMCOL') 

    #     A replace the missing values by np.nan
        df_gen = self.replace_cols_nan() 

    #     B get infos about the column nans - mark all columns with skip=True if the missing rate 
    #     is greater than 0.25
        cols_dic_nan = self.get_cols_nan( df_gen, p_missing_rate, p_set_skip )
        cols_df = pd.DataFrame.from_dict(cols_dic_nan, orient='index')
        cols_df = cols_df.query('skip == True') 
        if self.me_print == True:
            print('Columns to delete - skip=True\n', cols_df.shape, '\n')
            display(cols_df.head(20))

    #   C add manual collected columns to remove 
        cols_dic_nan = self.add_cols_2_remove( cols_dic_nan )

    #   D plot a bar chart for all columns which are marked as skip=True (to remove)
        print('meplot', self.me_plot)
        if self.me_plot == True:
            self.plot_cols_nan(df_gen, cols_dic_nan, self.master_dic)

    #   E Remove the outlier columns from the dataset. (You'll perform other data
    #   engineering tasks such as re-encoding and imputation later.)
    #   remove all columns marked as deletable 
        df_cols_notnan = self.del_cols_nan( df_gen, cols_dic_nan ) 
        return df_cols_notnan  

# ----------------------------------------------------------------
# Execution - data cleaning - nan handling for rows 
# Identify missing or unknown data values and convert them to NaNs.
# ----------------------------------------------------------------
    def do_row_nans( self, p_df, p_max_nan_columns ):

      self.print_progress('PRE_REMROW') 
      # How much data is missing in each row of the dataset? calculate new
      # if p_max_nan_columns == 0:
      df_rows_stat, max_nan_columns = self.get_rows_nan_stat(p_df)
      df_rows_notnan, df_rows_nan = self.get_rows_nan(p_df, p_max_nan_columns)
      return df_rows_notnan, max_nan_columns   

# -------------------------------------------------------------------
# replace all the values which are marked as missing values in the 
# column dictionary with np.nan
# -------------------------------------------------------------------
    def replace_cols_nan( self ):
      """
      # replace all the values which are marked as missing values in the 
      # column dictionary with np.nan
      """
      for col_dic in self.cols_dic.values():
        missing_list = col_dic['unknown_list']
        if len(missing_list) == 0:
            continue
        missing_int = []
        nan_list = []
        col_name = col_dic['col_name']
        col_type = self.df[col_name].dtype
        for missing_item in missing_list:
            if col_type == 'float64': 
                missing_int.append(float(missing_item))      
                nan_list.append(np.nan)
            else: 
                nan_list.append(np.nan)
        if len(missing_int) > 0:
            missing_list = missing_int
        self.df[col_name].replace(to_replace = missing_list, value = nan_list, 
                                inplace = True)
      return self.df  

# -------------------------------------------------------------------
# get all the columns with a certain rate of missing values and 
# mark them in the column dictionary as to remove ("SKIP")
# -------------------------------------------------------------------
    def get_cols_nan( self, p_df, p_nan_rate, p_set_skip ):      
        """
        # get all the columns with a certain rate of missing values and 
        # mark them in the column dictionary as to remove ("SKIP")
        """
        for col_dic in self.cols_dic.values():
            # print('item', col_dic)
            # missing_list = col_dic['missing_list'] 
            missing_list = [ np.nan ]
            col_name = col_dic['col_name']
            # search for the missing values
            missing_cnt = len(p_df.loc[p_df[col_name].isin(missing_list)]) 
            missing_rate =  missing_cnt / p_df.shape[0] 
            # do not change skip values for the customer dataset
            if p_set_skip == True:  
                if missing_rate > p_nan_rate:
                    missing_skip = True
                    missing_reason = 'high rate of NaNs'
                else: 
                    missing_skip = False
                    missing_reason = "" 
                self.cols_dic[col_name]['skip'] = missing_skip
                self.cols_dic[col_name]['skip_reason'] = missing_reason 
                    
            self.cols_dic[col_name]['unknown_cnt'] = missing_cnt
            self.cols_dic[col_name]['unknown_rate'] = missing_rate 
            # print('missing ', col_name, 'counter: ', missing_cnt, missing_rate )  
        return self.cols_dic 

# -------------------------------------------------------------------
# mark certain columns which should be removed in the column 
# dictionary as to remove (SKIP) together with a reason code
# -------------------------------------------------------------------
    def add_cols_2_remove( self, p_cols_dic ):
        """
        # mark certain columns which should be removed in the column 
        # dictionary as to remove (SKIP) together with a reason code
        """
    #     do not manually remove some columns - udacity expects that only 6 are removed
    #     p_cols_dic = upd_cols_skip('GEBAEUDETYP', 'diff.company/resedential not relevant', p_cols_dic)
        return p_cols_dic 

# -------------------------------------------------------------------
# helper to update the column dictionary
# -------------------------------------------------------------------
    def upd_cols_skip(self, p_col_name, p_reason, p_cols_dic ):
        """
        # helper to update the column dictionary
        """
        p_cols_dic[p_col_name]['skip_reason'] = p_reason  
        p_cols_dic[p_col_name]['skip'] = True 
        return p_cols_dic

# -------------------------------------------------------------------
# delete the columns of the dataframe which are marked as SKIP in the 
# columns dictionary 
# -------------------------------------------------------------------
    def del_cols_nan( self, p_df, p_cols_dic ):
        """
        # delete the columns of the dataframe which are marked as SKIP in the 
        # columns dictionary 
        """
        remove_cols = []
        for col_dic in p_cols_dic.values():
            col_name = col_dic['col_name']
            # remove column if marked as to skip 
            if col_dic['skip'] == True:
                if self.me_print == True: 
                    print('remove', col_name)
                remove_cols.append(col_name)
        if len(remove_cols) > 0:
          p_df_res = p_df.drop(axis = 1, columns = remove_cols )
        if self.me_print == True: 
            print('DataFrame after Column Deletion:', p_df_res.shape)
        return p_df_res

# -------------------------------------------------------------------
# plot a box plot of columsn to remove  
# -------------------------------------------------------------------
    def plot_cols_nan( self, p_df, p_cols_dic, p_master_dic): 
        """
        # plot a box plot of columns to remove  
        """
        plot_ind = 1
        plot_cnt = 0
        for col_item in p_cols_dic.values():
            col_name = col_item['col_name']
            if col_item['unknown_rate'] == 1.0: 
                print('Empty Column:', col_name)
                continue
            # print column if marked as to skip 
            if col_item['skip'] == True:
                plot_cnt += 1
                if plot_cnt > 10: 
                    break 
                plt.figure(figsize=(10,5))
                # plot a bar chart for the column
                ax = sns.countplot(data=p_df, x = col_name, color = GC_BASE_COLOR)
                plot_ind += 1
                plt.xticks(rotation=30, ha='right', fontsize=8)
                x_labels = ax.get_xticklabels() # get the currenct ticklabels
                for x_label in x_labels:
                    # get the currenct label (caution: 1 -> 1.0)
                    x_label_text = x_label.get_text().split('.')[0]
                    # prepare the key for the master data lookup 
                    x_label_key = col_name + '|' + x_label_text
                    # get the description for the master data key 
                    x_label_value = p_master_dic.get(x_label_key)
                    # add the key to the description
                    x_label_value = str(x_label_value) + ' / ' + x_label_text
                    # x_label_value = x_label_text 
                    if x_label_value != '':
                        x_label.set_text(x_label_value)
                ax.set_xticklabels(x_labels)
                plt.title(col_name + ' - ' + col_item['skip_reason'] ) 
        plt.show() 

# -------------------------------------------------------------------
# distribution of the no of nan columns per rows 
# col_max - col. no where 10% of the nan rows are affected   
# -------------------------------------------------------------------
    def get_rows_nan_stat(self, p_df):
        """ 
        calculate how many columns of a row has NaN values (absolute, relative, cumulated)
        """
        col_name = 'NaNColumns'
        # sums of nan columns 
        df_col_nan = pd.DataFrame(p_df.loc[:, :].isnull().T.sum())
        df_col_nan.columns = [col_name]
        # no of abolute observations per missing column number
        df_sum = df_col_nan.groupby(col_name)[col_name].count().to_frame() 
        df_sum.columns = ['cnt_abs']
        # percentage of observations per missing column number
        df_sum_rel = (df_col_nan.groupby(col_name)[col_name].count() / df_col_nan.shape[0]).to_frame()
        # cumulated percentage of observations per missing column number
        df_sum_cum = pd.DataFrame(df_sum_rel[col_name].cumsum(axis=0))
        df_sum_rel.columns = ['cnt_rel']
        df_sum_cum.columns = ['cnt_relcum']
        # put all together
        df_sum = df_sum.merge(df_sum_rel, on=col_name) 
        df_sum = df_sum.merge(df_sum_cum, on=col_name)
        # search where the cum value is > 90% and take the column no. at that point  
        col_max = (df_sum.query('cnt_relcum > 0.9')).index[0]
        return df_sum, col_max

# -------------------------------------------------------------------
# split the rows by how many empty columns are found   
# -------------------------------------------------------------------
    def get_rows_nan(self, p_df, p_max_nan_columns):
        """
        # split the rows by how many empty columns are found   
        """
        # series of rows with the no of columns with NaN values
        df_nan_sum = p_df.loc[:, :].isnull().T.sum()
        # select all rows with NaN columns > max 
        df_nan = df_nan_sum[df_nan_sum > p_max_nan_columns]
        # select all rows with NaN columsn < max
        df_notnan = df_nan_sum[df_nan_sum <= p_max_nan_columns]
        df_test = df_nan_sum[df_nan_sum <= p_max_nan_columns].to_frame() 
        # print('get_rows_nan', df_nan.shape, df_notnan.shape, df_nan_sum.shape, p_df.shape)
        # print('df', p_df['LNR'].head(10))
        # print('dfc', df_test.shape, df_test.columns, df_test.head())
        # print('nan', df_notnan.index[1:10] , len(df_notnan)) 
        # for ind in df_test.index:
        #     if ind not in p_df.index: 
        #         print('ind', ind )
        # select the rows of the dataframe with NaN columns > max
        # out of index problem when using iloc (index > array length)
        # df_rows_notnan = p_df.iloc[df_notnan.index]
        # df_rows_nan = p_df.iloc[df_nan.index]
        df_rows_notnan = p_df[p_df.index.isin(df_notnan.index)] 
        df_rows_nan = p_df[p_df.index.isin(df_nan.index)] 
        # df_rows_notnan = p_df.loc( p_df['LNR'] == df_notnan['LNR']) 
        # df_rows_notnan = pd.merge(p_df, df_notnan, how='inner', on=['LNR'])
        # print('df nan', df_rows_nan.shape)
        # print('df nan', df_rows_notnan.shape)
        print('Rows NaN: ',  len(df_rows_nan), ' Rows not-Nan: ', len(df_rows_notnan))

        return df_rows_notnan, df_rows_nan  

# -------------------------------------------------------------------
# check which column cannot be imputed    
# -------------------------------------------------------------------
    @classmethod
    def check_imputing(cls, p_df, p_col_name, p_strategy='mean'):
        """
        check which column cannot be imputed because of non-numeric value
        """
        impute_obj = SimpleImputer(missing_values=np.nan, strategy=p_strategy)
        for col in p_df.columns:
            print('column', col)
            # make a 2dim dataframe out of the single column values
            df_col = pd.DataFrame(data=np.hstack((p_df[col])))
            impute_model = impute_obj.fit(df_col)
    
# -------------------------------------------------------------------
# do multi category imputing - pca preparation   
# -------------------------------------------------------------------
    @classmethod
    def do_imputing_most(cls, p_df, p_strategy, p_models_dic): 
        """
        Use the SingleImputer to impute columns 
        p_strategy: most_frequent (categorial) or mean
        """
        cls.print_progress_c('PRE_IMPUTE') 
        if p_models_dic['m_impmost'] == '':
            impute_obj = SimpleImputer(missing_values=np.nan, strategy=p_strategy)
            impute_model = impute_obj.fit(p_df)
            p_models_dic['m_impmost'] = impute_model
        else:
            impute_model = p_models_dic['m_impmost'] # p_impute_model 
        df_imputed = pd.DataFrame(impute_model.transform(p_df), columns=p_df.columns)
        return df_imputed, p_models_dic

# -------------------------------------------------------------------
# do pca category imputing - pca preparation   
# -------------------------------------------------------------------
    @classmethod
    def do_imputing_mean(cls, p_df, p_strategy, p_models_dic): 
        """
        Use the SingleImputer to impute columns 
        p_strategy: most_frequent (categorial) or mean
        """
        cls.print_progress_c('PRE_IMPUTE') 
        if p_models_dic['m_impmean'] == '':
            impute_obj = SimpleImputer(missing_values=np.nan, strategy=p_strategy)
            impute_model = impute_obj.fit(p_df)
            p_models_dic['m_impmean'] = impute_model
        else:
            impute_model = p_models_dic['m_impmean'] # p_impute_model 
        df_imputed = pd.DataFrame(impute_model.transform(p_df), columns=p_df.columns)
        return df_imputed, p_models_dic

# -------------------------------------------------------------------
# reencode 
# onehot encoding for categorial columns with many values 
# -------------------------------------------------------------------
    @classmethod
    def do_reencode_multi(cls, p_df, p_cols_dic,  p_models_dic, p_print ):
        """
        # reencode 
        # onehot encoding for categorial columns with many values 
        """
        df_cols_dic = pd.DataFrame.from_dict(p_cols_dic, orient='index')
        # columns of type categories with multiple values 
        df_cols_dic_multi = df_cols_dic[(df_cols_dic['datatype'] == 'categorical') & (df_cols_dic['skip'] == False )
                                                & (df_cols_dic['unique_cnt'] > 2 )]
        if p_print == True:
            print('Multi', df_cols_dic_multi.shape)
        # all the columns 
        cols_multi = list(df_cols_dic_multi['col_name'])
        # column labels
        cols_label_multi = [] 
        for ind, row in df_cols_dic_multi.iterrows(): 
            col_name = row['col_name']
            for val in p_df[col_name].dropna().unique():
                if type(val) is np.float64:   
                    val = int(val) 
                cols_label_multi.append(col_name + '_' + str(val)) 
        # print(cols_label_multi)
        # replace the nan values by the mode value
        df_multi_imputed, p_models_dic = cls.do_imputing_most(p_df[cols_multi], 
                                        'most_frequent', p_models_dic) 

        # onehot encoding
        # df_multi_dummies = pd.get_dummies(df_multi_imputed)
        if p_print == True:     
            print('Before OneHotEncoded: ', p_df.shape)
        #if p_models_dic['m_onehot'] == '': 
        onehot_model = OneHotEncoder().fit(df_multi_imputed)
        #    p_models_dic['m_onehot'] = onehot_model
        # else:
        #    onehot_model = p_models_dic['m_onehot']

        # print('Model OneHotEncoded:', model_onehot) 
        df_multi_onehot = pd.DataFrame((onehot_model.transform(df_multi_imputed)).toarray(), 
                                      columns=cols_label_multi)
        # remove the old columns 
        p_df.drop(cols_multi, axis = 1, inplace = True)
        p_df.reset_index(inplace = True, drop = True )
        # add the new encoded columns
        p_df = pd.concat([p_df, df_multi_onehot], axis = 1) 
        if p_print == True:    
            print('After OneHotEncoded: ', p_df.shape) 

        return p_df, df_multi_onehot, p_models_dic 