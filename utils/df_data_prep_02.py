import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import math 
from datetime import datetime
from pytz import timezone

from sklearn.impute import SimpleImputer

from utils.df_data_prep_01 import DfDataPrep01

# -------------------------------------------------------------------
# constants  
# -------------------------------------------------------------------
GC_MAX_ROWS = 1000
GC_MAX_MISSING_RATE = 0.25
GC_UTC = timezone('UTC')

class DfDataPrep02: 
    """ 
    class to do some project specific data preparation 
    """
    
    def __init__(self, p_df, p_cols_dic, p_models_dic, p_print=True, p_plot=True ):
        """
        Instantiates the object with the dataset 
        Input: 
        - Dataset to analyze
        """
        self.df = p_df
        self.cols_dic = p_cols_dic 
        self.models_dic = p_models_dic 
        self.set_print(p_print)
        self.set_plot(p_plot) 

    def set_print(self, p_print): 
        self.me_print = p_print 

    def set_plot(self, p_plot): 
        self.me_plot = p_plot 

# -------------------------------------------------------------------
    def print_progress(self, p_me, p_title=''):
        """
        print a tile together with a time stamp in a formatted way
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
        # elif p_me == 'TRAIN_CLEAN':
        #     print(tmst, ' Mailout Train Clean Data ----------------------')
        # elif p_me == 'TRAIN_PCA': 
        #     print(tmst, ' Mailout Train PCA -----------------------------')
        # elif p_me == 'TRAIN_KMEANS': 
        #     print(tmst, ' Mailout Train KMeans --------------------------')
        # elif p_me == 'TRAIN_FIT': 
        #     print(tmst, ' Mailout Train Fit -----------------------------')
        # elif p_me == 'TRAIN_PREDICT': 
        #     print(tmst, ' Mailout Train Predict -------------------------')
        elif p_me == 'READY': 
            print(tmst, ' Ready -----------------------------------------')
        elif p_me == 'FREE':
            line = ' -----------------------------------------------'
            rest_len = len(line) - len(p_title) 
            header = p_title + line[:rest_len]
            print(tmst, header) 

    def do_reencode(self, p_df): 

        """
        triggers all the necessary reencodings relevant for the project
        Input: 
        - p_df: data frame which should be reencoded 
        """ 
        # select, re-encode, and engineer column values.
        self.print_progress('PRE_REENCODE')
        df_prep = self.do_reencode_ostwest_cameodeu(p_df) 
        df_prep = self.do_reencode_year(df_prep) 
        df_prep, df_onehot, self.models_dic = DfDataPrep01.do_reencode_multi(df_prep, 
                self.cols_dic, self.models_dic, self.me_print )
        df_prep = self.do_reencode_pj_jugend(df_prep)
        df_prep = self.do_reencode_cameo(df_prep )

        return df_prep, self.models_dic 

# -------------------------------------------------------------------
# reencode some columns 
# replace characters by integer 
# -------------------------------------------------------------------
    def do_reencode_ostwest_cameodeu(self, p_df):
        """
        # reencode some columns (OST_WEST_KZ, CAMEO_DEUG_2015, CAMEO_DEU_2015)
        # replace characters by integer 
        """
        col_name = 'OST_WEST_KZ'
        if col_name in p_df.columns:
            col_old = p_df[col_name].unique() 
            p_df[col_name].replace(to_replace={'W': 0, 'O': 1}, inplace = True)
            if self.me_print == True:  
                print(col_name, '--------------------')
                print('old', col_old) 
                print('new', p_df[col_name].unique()) 

        col_name = 'CAMEO_DEUG_2015' 
        if col_name in p_df.columns:
            if p_df[col_name].dtype != 'float64':
                col_old = p_df[col_name].unique() 
                p_df[col_name].replace(to_replace={'X': np.nan, '1': 1, '2': 2, '3': 3, 
                                '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}, inplace = True)
            if self.me_print == True:  
                print(col_name)
                print('old', col_old) 
                print('new', p_df[col_name].unique()) 
                
        col_name = 'CAMEO_DEU_2015' 
        if col_name in p_df.columns:
            col_old = p_df[col_name].unique() 
            p_df[col_name] = p_df[col_name].str.replace('XX', '0')
            p_df[col_name] = p_df[col_name].str.replace('A', '1')
            p_df[col_name] = p_df[col_name].str.replace('B', '2')
            p_df[col_name] = p_df[col_name].str.replace('C', '3')
            p_df[col_name] = p_df[col_name].str.replace('D', '4')
            p_df[col_name] = p_df[col_name].str.replace('E', '5')
            p_df[col_name] = p_df[col_name].str.replace('F', '6')
            if self.me_print == True:  
                print(col_name, '--------------------')
                print('old', col_old) 
                print('new', p_df[col_name].unique()) 
        return p_df 

# -------------------------------------------------------------------
# reencode the PRAEGENDE_JUGENDJAHRE column 
# split in new columns for decade and movement
# -------------------------------------------------------------------
    def do_reencode_pj_jugend(self, p_df):
        """
        # reencode the PRAEGENDE_JUGENDJAHRE column 
        # split in new columns for decade and movement
        """
        col_name = 'PRAEGENDE_JUGENDJAHRE'
        if col_name not in p_df.columns:
            return p_df   

        # Jugendjahre
        conditions= [
            (p_df[col_name] >= 1 ) & (p_df[col_name] <= 2 ), # 40s
            (p_df[col_name] >= 3 ) & (p_df[col_name] <= 4 ), # 50s
            (p_df[col_name] >= 5 ) & (p_df[col_name] <= 7 ), # 60s
            (p_df[col_name] >= 8 ) & (p_df[col_name] <= 9 ), # 70s
            (p_df[col_name] >= 10 ) & (p_df[col_name] <= 13 ), # 80s
            (p_df[col_name] >= 14 ) & (p_df[col_name] <= 15 ) # 90s
        ]
        values = [
            4, 5, 6, 7, 8, 9
        ]
        p_df['PJ_DECADE'] = np.select(conditions, values) 
        # Movement
        cond_move = [
            (p_df[col_name] == 1 ), (p_df[col_name] == 3 ), # maintream
            (p_df[col_name] == 5 ), (p_df[col_name] == 8 ),  
            (p_df[col_name] == 10 ), (p_df[col_name] == 12 ),  
            (p_df[col_name] == 14 )
        ]
        value_move = [
            1, 1, 1, 1, 1, 1, 1
        ]
        p_df['PJ_MOVEMENT'] = np.select(condlist = cond_move, choicelist = value_move, 
                                        default=0) 

        if self.me_print == True:  
            print(col_name, '--------------------')
            print('old', p_df[col_name].unique())
            print('new decade', p_df['PJ_DECADE'].unique()) 
            print('new movement', p_df['PJ_MOVEMENT'].unique())
        # print(p_df['PJ_MOVEMENT'].value_counts())
        # print(p_df[col_name].value_counts())

        # drop the original column 
        p_df.drop(axis = 1, columns = [col_name], inplace = True)  

        return p_df 

# -------------------------------------------------------------------
# reencode the CAMEO_INTL_2015 column 
# split in new columns for wealth and life staged
# -------------------------------------------------------------------
    def do_reencode_cameo(self, p_df):
        """
        # reencode the CAMEO_INTL_2015 column 
        # split in new columns for wealth and life staged
        """
        col_name = 'CAMEO_INTL_2015'
        if col_name not in p_df.columns:
            return p_df 
 
        if p_df[col_name].dtype != 'float64':
            p_df[col_name].replace(to_replace={'XX': 99}, inplace = True)
        p_df[col_name].replace(to_replace={np.nan: 99}, inplace = True)
        p_df['CAMEO_LIFESTAGE'] = p_df[col_name].astype(int) // 10 
        p_df['CAMEO_LIFESTAGE'].replace(to_replace={9: np.nan}, inplace = True)
        p_df['CAMEO_WEALTH'] = p_df[col_name].astype(int) % 10 
        p_df['CAMEO_WEALTH'].replace(to_replace={9: np.nan}, inplace = True)
        if self.me_print == True:  
            print(col_name, '--------------------')
            print('old', p_df[col_name].unique())
            print('new wealth', p_df['CAMEO_WEALTH'].unique()) 
            print('new lifestage', p_df['CAMEO_LIFESTAGE'].unique()) 

        # drop the original column 
        p_df.drop(axis = 1, columns = [col_name], inplace = True)  
        return p_df 

    def do_reencode_year(self, p_df ):
        """
        # reencode the EINGEFUEGT_AM column 
        # split in new columns for the year 
        """
        col_name = 'EINGEFUEGT_AM'
        if col_name not in p_df.columns:
            return p_df  
        
        p_df['INSERT_YEAR'] = p_df[col_name].str.slice(0, 4)
        if self.me_print == True:  
            print(col_name, '--------------------')
            # print('old', p_df[col_name].unique())
            print('new year', p_df['INSERT_YEAR'].unique()) 
            
        # drop the original column 
        p_df.drop(axis = 1, columns = [col_name], inplace = True)  
        return p_df 