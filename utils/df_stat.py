import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns 
import math 
from datetime import datetime
from pytz import timezone
from IPython.display import display

# -------------------------------------------------------------------
# constants  
# -------------------------------------------------------------------
GC_MAX_ROWS = 1000
GC_MAX_MISSING_RATE = 0.25
GC_UTC = timezone('UTC')
GC_BASE_COLOR = sns.color_palette()[0]
GC_COLORMAP = plt.cm.RdBu

class DfStat: 
    """ 
    class with some helper methods to print statistics around a dataset 
    """
    
    def __init__(self, p_df, p_cols_dic):
        """
        Instantiates the object with the dataset 
        Input: 
        - Dataset to analyze
        """
        self.df = p_df
        self.df_col_dic = p_cols_dic 

# -------------------------------------------------------------------
# get all the columns out of the feature list and extract some info
# about the missing values and the column data type
# -------------------------------------------------------------------
    # def get_cols_dic( self, p_df, p_feat, p_attr, p_nan_rate=0.5, p_print='' ): 
    #     """
    #     # get all the columns out of the feature list and extract some info
    #     # about the missing values and the column data type
    #     """
    #     cols_type = p_df.dtypes
    #     cols_dic = {}
    #     row_cnt = p_df.shape[0]
    #     feat_unknown = p_attr.query('Meaning == "unknown"')
    #     feat_numeric = p_attr.query('Meaning == "numeric value"')
    #     for col_name in p_df.columns:
    #         # init's
    #         col_skip = ''
    #         col_skip_reason = '' 
    #         col_desc = ''
    #         col_not_in_dic = ''
    #         col_unknown = '' 
    #         col_datatype = '' 
    #         col_unknown_rate = 0.0
    #         col_unknown_cnt = 0.0
    #         col_unknown_list = []
    #         col_unknown_skip = False 
    #         # calculation
    #         col_type = cols_type[col_name]
    #         col_int = True if col_type == 'int64' or col_type == 'float64' else False 
    #         col_nan  = p_df[col_name].isna().sum()
    #         col_nan_ratio = col_nan / row_cnt 
    #         col_unique_cnt = p_df[col_name].nunique()
    #         col_unique = p_df[col_name].unique()[:5]
    #         if col_name not in p_feat.index: 
    #             col_not_in_dic = 'X'
    #             if p_print == 'PRINT':
    #                 print('column not in feature dictionary', col_name)
    #         if col_name in feat_unknown.index: 
    #             col_unknown = feat_unknown.loc[col_name].Value
    #         col_unknown_cnt, col_unknown_rate, col_unknown_list \
    #           = self.get_cols_missing( p_df, col_name, col_unknown, col_int)
    #         if col_name in p_feat.index: 
    #             col_desc = p_feat.loc[col_name].Description
    #             col_datatype = p_feat.loc[col_name].DataType
    #         if col_name in feat_numeric.index: 
    #             col_datatype = 'numeric'
    #         if col_type == 'object':
    #             col_datatype = 'categorial'
    #         if col_unknown_rate > p_nan_rate:
    #             col_unknown_skip = True
    #             col_skip_reason = 'high rate of NaNs'
    #             col_skip = True 
    #         cols_dic[col_name] = {'col_name': col_name, 'type': col_type , 'datatype': col_datatype, 
    #                               'inttype': col_int, 'skip': col_skip, 'desc': col_desc, 
    #                             'nan_cnt': col_nan, 'nan_ratio': col_nan_ratio,
    #                             'unique_cnt': col_unique_cnt, 'unique': col_unique,
    #                             'unknown': col_unknown, 'unknown_cnt': col_unknown_cnt,
    #                             'unknown_rate': col_unknown_rate, 
    #                             'unknown_skip': col_unknown_skip,
    #                             'unknown_list': col_unknown_list, 
    #                             'skip_reason': col_skip_reason, 
    #                             'not_in_dic': col_not_in_dic } 
    #     return cols_dic

# -------------------------------------------------------------------
# convert column dictionary for certain columns into a dataframe
# -------------------------------------------------------------------
    # def get_cols_dic_as_df(self, p_dic, p_cols):
    #     """
    #     convert into column dictionary into dataframe
    #     input: p_dic dictionary
    #           p_cols columns to export 
    #     export: column dictionary as dataframe  
    #     """
    #     if len(p_cols) == 0: 
    #         cols = ['type', 'skip', 'not_in_dic', 'nan_cnt', 'nan_ratio', 
    #                 'unique_cnt', 'unknown_cnt', 'unknown_rate', 'unique']
    #     else:
    #         cols = p_cols
    #     df_cols_dic = pd.DataFrame.from_dict(p_dic, orient='index')
    #     return df_cols_dic[cols]


    # def get_cols_missing( self, p_df, p_col_name, p_col_unknown, p_col_int ): 
    #     missing_list = [] 
    #     missing_int = []
    #     missing_cnt = 0
    #     missing_rate = 0.0
    #     if len(str(p_col_unknown)) > 0: 
    #         # convert string to list 
    #         # missing_list = list(p_col_unknown.values[0].replace('[', '').replace(']', '').split(','))
    #         # convert search string to integer if necessary
    #         missing_list = list(str(p_col_unknown).split(','))
    #         if p_col_int == True:  
    #             for item in missing_list:
    #                 if item == 'X' or item == 'XX': item = '-1'
    #                 if item == 'XX': item = '-1'
    #                 # print('col', col_name, col_int )
    #                 if item != '':
    #                     missing_int.append(int(item))
    #                 missing_list = missing_int 
    #     missing_list.append(np.nan)
    #     # info about the missed values 
    #     missing_cnt = len(p_df.loc[p_df[p_col_name].isin(missing_list)]) 
    #     missing_rate =  missing_cnt / p_df.shape[0] 
    #     return missing_cnt, missing_rate, missing_list

    def get_df(self):
        """ standard getter to get the dataset"""
        return self.df 
    
    def set_df(self, p_df):
        """ standard setter to change the dataset"""
        self.df = p_df

    def set_print_options(self, p_me = 'DEFAULT'):
        """ 
        set print options to allow printing all rows and columns
        Input: 
        - p_me : ALL = print all rows/columns
        """
        if p_me == 'ALL':
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_colwidth',60) 
            pd.set_option('display.max_columns', None)
        else:
            pd.set_option('display.max_rows', 0)
            pd.set_option('display.max_colwidth',0) 
            pd.set_option('display.max_columns', 0) 

# -------------------------------------------------------------------
    def print_progress(p_me, p_title=''):
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

    def print_shape(self):
        """ print the shape of the dataset"""
        print(self.df.shape)

    def print_no(self):
        """
        Print some useful numbers about the dataset
        """
        tmst =  datetime.now(GC_UTC)
        print('--------------------------------------------------')
        print('Numbers ', tmst )
        print('--------------------------------------------------')
        print('Number of rows {:,} and columns {:,}'.format(self.df.shape[0], self.df.shape[1]))
        print('Number of NaN values in our DataFrame:', self.df.isnull().sum().sum())
        print('Number of columns with NaN values:', self.df.isnull().any().sum())
        col_no = int(self.df.shape[1] / 10)
        print('Number of rows with NaN columns > {:,}:'.format(col_no), 
                (self.df.loc[:, :].isnull().T.sum().values > col_no).sum())

    def print_desc(self):
        """
        Use the describe method of the dataset to print some statistics
        """
        tmst =  datetime.now(GC_UTC)
        print('--------------------------------------------------')
        print('Column Statistics DESC' + tmst )
        print('--------------------------------------------------\n')
        print(str(self.df.describe().T))

    def get_cluster_sum(self, p_df): 
        df_sum = p_df.groupby(['cluster_no', 'RESPONSE'])['RESPONSE'].count().to_frame() 
        df_sum.columns = ['cnt']
        df_sum_clus = df_sum.unstack(level=-1) 
        # remove last multi-index level 
        df_sum_clus.columns = df_sum_clus.columns.droplevel(-1) 
        df_sum_clus.columns = ['cnt_0', 'cnt_1']
        df_sum_clus['sum'] = df_sum_clus.iloc[:,0] + df_sum_clus.iloc[:,1] 
        df_sum_clus['ratio_sum'] = df_sum_clus.iloc[:,1] / df_sum_clus.iloc[:,2] 
        df_sum_clus.sort_values(by=['ratio_sum'], ascending=False,inplace=True)
        df_sum_clus['ratio_1'] = df_sum_clus.iloc[:,1] / df_sum_clus.iloc[:,1].sum()
        df_sum_clus['sum_1_cum'] = df_sum_clus.iloc[:,3].cumsum()
        return df_sum_clus

    def print_cluster_sum(self, p_df, p_cluster_no=20):
        df_sum = self.get_cluster_sum(p_df) 
        print('sum', df_sum.shape)
        display(df_sum.head(p_cluster_no))

    def plot_cluster_sum(self, p_df): 
        df_sum = self.get_cluster_sum(p_df).sort_values(by='cluster_no')
        fig, axs = plt.subplots(1,2)

        # y = df_sum[['cnt_1']].groupby(['cluster_no']).sum()
        # y = y.iloc[:,0]
        y = df_sum['cnt_1']
        x = df_sum.index.values.astype(int).tolist() 
        # pie chart - on the right
        axs[1].pie( y, labels=x, autopct='%1.1f%%' )
        axs[1].set_title("Reponse per Cluster - Ratio")
        # bar chart - on the left 
        axs[0].bar( x, y )
        axs[0].set_title("Respose per Cluster - Sum")

        plt.xlabel('Cluster No')
        plt.xticks(ha='right', fontsize=8)
        # format without decimals 
        formatter = ticker.FormatStrFormatter('%2.0lf')
        axs[0].xaxis.set_major_formatter(formatter)
        # x-ticks + x-label - avoiding shifts 
        for x_item, y_item in zip(x, y):    
            axs[0].annotate(
                str(y_item),        # label is our y-value as a string
                xy=(x_item, y_item),
                xytext=(0,3), # 3 points vertical offset
                textcoords="offset points",
                ha='center', 
                va='bottom'
            )
        axs[0].set_xticks(x)
        plt.ylabel('No of Responses')
        plt.show() 

    def print_col_distinct(self, p_col_name):
        """
        Print the distinct values of a column
        Input: 
        - p_col_name: column name
        """
        tmst =  datetime.now(GC_UTC)
        print('--------------------------------------------------')
        print('Column Distinct Values of ', p_col_name, ' ', tmst)
        print('--------------------------------------------------\n')
        print(self.df[p_col_name].value_counts() )
        print('Total No.:', self.df[p_col_name].nunique())

    def print_info(self):
        tmst =  datetime.now(GC_UTC)
        print('--------------------------------------------------')
        print('Column Statistics INFO', tmst)
        print('--------------------------------------------------\n')
        print(str(self.df.info(show_counts=True)))

# -------------------------------------------------------------------
# print some statistic data depending on the chosen method P_ME 
# -------------------------------------------------------------------
    def print_stat(self, p_me, p_df):
        """
        # print some statistic data depending on the chosen method P_ME 
        """
        self.set_print_options('ALL')
      
        if p_me == 'NO' or p_me == 'ALL':
            print('--------------------------------------------------')
            print('Numbers NO')
            print('--------------------------------------------------')
            print('Number of rows {:,} and columns {:,}'.format(p_df.shape[0], p_df.shape[1]))
            print('Number of NaN values in our DataFrame:', p_df.isnull().sum().sum())
            print('Number of columns with NaN values:', p_df.isnull().any().sum())
            col_no = int(p_df.shape[1] / 10)
            print('Number of rows with NaN columns > {:,}:'.format(col_no), 
                (p_df.loc[:, :].isnull().T.sum().values > col_no).sum())
        
        if p_me == 'TI' or p_me == 'ALL':
            print('--------------------------------------------------')
            print('Column Titles TI')
            print('--------------------------------------------------\n')
            print(p_df.columns)

        if p_me == 'TYN' or p_me == 'ALL':
            print('--------------------------------------------------')
            print('Column Type Numbers TYN')
            print('--------------------------------------------------\n')
            print(p_df.dtypes.value_counts())

        if p_me == 'TY' or p_me == 'ALL':
            print('--------------------------------------------------')
            print('Column Types TY')
            print('--------------------------------------------------\n')
            print(p_df.dtypes)

        if p_me == 'VA' or p_me == 'ALL':
            # no of non-NaN values in our DataFrame 
            print('--------------------------------------------------')
            print('Columns with values VA')
            print('--------------------------------------------------\n')
            print(p_df.count())

        if p_me == 'DESC' or p_me == 'ALL':
            print('--------------------------------------------------')
            print('Column Statistics DESC')
            print('--------------------------------------------------\n')
            print(str(p_df.describe().T))

        # type + non-nan values per column
        if p_me == 'INFO' or p_me == 'ALL':
            print('--------------------------------------------------')
            print('Column Statistics INFO')
            print('--------------------------------------------------\n')
            print(str(p_df.info(show_counts=True)))

        self.set_print_options()

# -------------------------------------------------------------------
# print the column dictionary 
# -------------------------------------------------------------------
    def print_cols_dic( self, p_cols_dic, p_numrec=5):  
        """
        # print the column dictionary 
        """
        self.set_print_options('ALL')
        cols_df = pd.DataFrame.from_dict(p_cols_dic, orient='index')
        cols_df = cols_df.sort_values(by=["col_name"])
        print('--------------------------------------------------')
        print('Column Dictionary')
        print('--------------------------------------------------\n')
        print(str(cols_df.head(p_numrec)))
        # print(cols_df.to_markdown())
        # print(cols_df.head(p_numrec))
        self.set_print_options()

# -------------------------------------------------------------------
# print general/customer compare  
# -------------------------------------------------------------------
    def print_colcompare(self, p_df_1, p_df_2, p_add, p_print=True):
        """
        # print column compare betweeen 2 data frames  
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        if p_print == True:
            print('Gen/Customer', p_df_1.shape, p_df_2.shape) 
        for col in p_df_1.columns: 
            if col not in  p_df_2.columns:
                if p_add == True:
                    col_mean = p_df_1[col].mean() 
                    p_df_2[col] = col_mean 
                if p_print == True: 
                    print('General Column without Customer Column', col) 
        for col in p_df_2.columns:
            if col not in p_df_1.columns:
                if p_print == True: 
                    print('Customer Column without General Column', col)
                if p_add == True:
                    p_df_2.drop(axis = 1, columns = [col], inplace = True)  
        return p_df_1, p_df_2

# -------------------------------------------------------------------
# plot a box plot of columns   
# -------------------------------------------------------------------
    def plot_cols(self, p_df, p_cols, p_master_data): 
        """
        # plot a box plot of columns   
        """
        plot_ind = 1
        for col in p_cols:
            col_name = col 
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
                x_label_value = p_master_data.get(x_label_key)
                # add the key to the description
                x_label_value = str(x_label_value) + ' / ' + x_label_text
                if x_label_value != '':
                    x_label.set_text(x_label_value)
            ax.set_xticklabels(x_labels)
            plt.title(col_name ) 
        plt.show() 

# -------------------------------------------------------------------
# plot a histogram for every column in the parameter p_col_series 
# -------------------------------------------------------------------
    def plot_col_hist(p_df, p_col_series):
        """
        # plot a histogram for every column in the parameter p_col_series 
        """
        p_col_series.sort()
        col_no = len(p_col_series)
        row_re = col_no % 3
        row_no = math.floor(col_no / 3) 
        if row_re > 0: row_no += 1 
        if col_no > 3: col_no = 3
        plot_ind = 1
        plot_height = row_no * 4
        # plt.clf() 
        plt.figure(figsize=(10,plot_height) )
        # plt.figure()
        # plt.subplots_adjust(top=0.85)
        # plt.tight_layout() 
        # plt.style.use('seaborn')
        
        # plt.xticks(rotation = 45)
        for col in p_col_series:
            colors = ['blue'] # ['yellowgreen','cyan','magenta']
            ax = plt.subplot(row_no, col_no, plot_ind)
            ax.set_title(col, fontsize=8)
            ax.set_xlabel('')
            fig = ax.get_figure()
            fig.subplots_adjust(top=1.4)
            fig.tight_layout()
            # ax.set_title(col)  
            p_df.groupby(col)[col].count().plot(kind='bar', color=colors)
            # p_df.hist(column=[col])
            plot_ind += 1
        plt.show() 

       
# -------------------------------------------------------------------
# plot a box plot of columsn to remove  
# -------------------------------------------------------------------
    def plot_cols_nan( p_df, p_cols_dic, p_master_data): 
        """
        # plot a box plot of columsn to remove  
        """
        plot_ind = 1
        for col_item in p_cols_dic.values():
            col_name = col_item['col_name']
            # print column if marked as to skip 
            if col_item['skip'] == True:
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
                    x_label_value = p_master_data.get(x_label_key)
                    # add the key to the description
                    x_label_value = str(x_label_value) + ' / ' + x_label_text
                    if x_label_value != '':
                        x_label.set_text(x_label_value)
                ax.set_xticklabels(x_labels)
                plt.title(col_name + ' - ' + col_item['missing_reason'] ) 
        plt.show()