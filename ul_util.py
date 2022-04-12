# import numpy as np 
# import pandas as pd
# from scipy.stats.stats import _validate_distribution
# from sklearn.preprocessing import StandardScaler 
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import Imputer
# import matplotlib.pyplot as plt 
# import seaborn as sns
# import os
# from pathlib import Path 
# import math
# from datetime import datetime
# from pytz import timezone

p_estimators=['LOGREG', 'RANFOR', 'XGB']  

      # train classifier 
ests = p_estimators
model_res = {}
for est in ests[0:1]:
    print('est', est)
exit() 



# -------------------------------------------------------------------
# constants  
# -------------------------------------------------------------------
gc_max_nan_columns = 5 
gc_base_color = sns.color_palette()[0]
gc_max_missing_rate = 0.18
gc_utc = timezone('UTC')

def print_progress(p_me):
    tmst =  datetime.now(gc_utc)
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
    elif p_me == 'GEN_PCA': 
        print(tmst, ' General PCA -----------------------------------')
    elif p_me == 'GEN_KMEANS': 
        print(tmst, ' General KMeans---------------------------------')
    elif p_me == 'CUS_LOAD': 
        print(tmst, ' Customer Load Data-----------------------------')
    elif p_me == 'CUS_CLEAN':
        print(tmst, ' Customer Clean Data ---------------------------')
    elif p_me == 'CUS_PCA': 
        print(tmst, ' Customer PCA ----------------------------------')
    elif p_me == 'CUS_KMEANS': 
        print(tmst, ' Customer KMeans -------------------------------')

# -------------------------------------------------------------------
# Perform feature trimming, re-encoding, and engineering for demographics
# data
# -------------------------------------------------------------------
def clean_data(p_df, p_feat, p_cols_dic, p_print):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ..
    print_progress('PRE_REMCOL') 
    cols_dic = {} 
    #  get infos about the column features
    if len(p_cols_dic) > 0:
        cols_dic = p_cols_dic
        set_skip = False # do not change skip value for customer dataset
    else:
        cols_dic = get_cols_dic( p_df, p_feat, gc_max_missing_rate )
        set_skip = True  
    # replace the missing values by np.nan
    df = replace_cols_nan(p_df, cols_dic) 
    # get infos about the column nans 
    cols_dic = get_cols_nan( df, cols_dic, gc_max_missing_rate, set_skip )
    # add manual collected columns to remove 
    cols_dic = add_cols_2_remove( cols_dic )
    
    # remove selected columns and rows, ...
    # remove all columns marked as deletable 
    df_prep = del_cols_nan( df, cols_dic, p_print) 

    print_progress('PRE_REMROW') 
    # How much data is missing in each row of the dataset?
    df_rows_notnan, df_rows_nan = get_rows_nan(df_prep)

    # select, re-encode, and engineer column values.
    print_progress('PRE_REENCODE')
    df_prep = do_reencode(df_rows_notnan, p_print) 
    df_prep, df_onehot = do_reencode_multi(df_prep, cols_dic, p_print)
    df_prep = do_reencode_pj_jugend(df_prep, p_print)
    df_prep = do_reencode_cameo(df_prep, p_print)

    # imputing 
    print_progress('PRE_IMPUTE')
    df_imputed = do_pca_imputing(df_prep, 'mean' ) 
    
    # Return the cleaned dataframe.
    return df_imputed, df_prep, cols_dic  

# -------------------------------------------------------------------
# set print options to allow printing all rows and columns  
# -------------------------------------------------------------------
def set_print_options(p_me = 'DEFAULT'):
    """
    # set print options to allow printing all rows and columns  
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
# get the kmeans score values for a target clsuter (n_comp)
# -------------------------------------------------------------------
def get_kmeans(cluster_no, p_df):
    """
    # get the kmeans score values for a target clsuter (n_comp)
    """
    k_score = 0
    k_cluster = []
    kmeans = KMeans(cluster_no)
    k_model = kmeans.fit(p_df)
    k_label = k_model.predict(p_df)
    k_center = k_model.cluster_centers_
    unique, counts = np.unique(k_label, return_counts = True )
    k_cluster = np.column_stack((unique, counts))
    k_score = np.abs(k_model.score(p_df))
    print('KMeans - Shape ', kmeans.cluster_centers_.shape, k_model.inertia_, 'No of Labels: ', len(k_label),
                           ' Score: ', k_score)
    # print('KMeans - Center', k_model.cluster_centers_)
    return k_score, k_cluster, k_center 

# -------------------------------------------------------------------
# get the original feature weights of a certain kmeans cluster 
# back from the center values 
# -------------------------------------------------------------------
def get_kmeans_inverse(p_pca, p_scaler, p_center, p_df):
    """
    # get the original feature weights of a certain kmeans cluster 
    # back from the center values 
    """
    pca_inverse = p_pca.inverse_transform(p_center)
    pca_scaled = p_scaler.inverse_transform(pca_inverse)
    overrep_df = pd.DataFrame(pca_scaled, index=p_df.columns, columns = ['Weight'])
    df_feat = overrep_df.sort_values(by = 'Weight', ascending = False)
    return df_feat

# -------------------------------------------------------------------
# get the kmeans score values for a number of chosen target
# clusters (n_comp)  
# -------------------------------------------------------------------
def get_kmeans_scores(cluster_nos, p_df):
    """
    # get the kmeans score values for a number of chosen target
    # clusters (n_comp)  
    """
    k_scores = []
    k_clusters = []
    k_centers = []
    cluster_no_max = 30
    for ind in cluster_nos: # range(1,cluster_no):
        k_score, k_cluster, k_center = get_kmeans(ind, p_df) 
        # no of distinct clusters found smaller than target cluster no -> stop  
        if len(k_cluster) > cluster_no_max: 
            break
        k_scores.append(k_score) 
        k_clusters.append(k_cluster) 
        k_centers.append(k_center)
        # print('Cluster:' , k_cluster.reshape(1,-1))
        # print('Score', ind, k_score)
    return k_scores, k_clusters, k_centers 

# -------------------------------------------------------------------
# split the rows by how many empty columns are found   
# -------------------------------------------------------------------
def get_rows_nan(p_df):
    """
    # split the rows by how many empty columns are found   
    """
    # series of rows with the no of columns with NaN values
    df_nan_sum = p_df.loc[:, :].isnull().T.sum()
    # select all rows with NaN columns > 5 
    df_nan = df_nan_sum[df_nan_sum > gc_max_nan_columns]
    # select all rows with NaN columsn < 5
    df_notnan = df_nan_sum[df_nan_sum <= gc_max_nan_columns]
    # select the rows of the dataframe with NaN columns > 5
    df_rows_notnan = p_df.iloc[df_notnan.index]
    df_rows_nan = p_df.iloc[df_nan.index]
    print('Rows NaN: ',  len(df_rows_nan), ' Rows not-Nan: ', len(df_rows_notnan))

    return df_rows_notnan, df_rows_nan  

# -------------------------------------------------------------------
# plot the kmeans scores - elbough plot   
# -------------------------------------------------------------------
# def plot_kmeans_scores(n_comp, p_scores):
#     centers = list(range(1,n_comp))
  
#     plt.plot(centers, p_scores)
#     plt.title('KMeans')
#     plt.xlabel('Centers')
#     plt.ylabel('Average Distance from Centroid')
#     plt.show() 

# -------------------------------------------------------------------
# do pca transformation - feature reduction  
# -------------------------------------------------------------------
def do_pca(n_comp, p_df): 
    """
    # do pca transformation - feature reduction  
    """
    pca_scaler = StandardScaler()
    X_scaled = pca_scaler.fit_transform(p_df)
    pca = PCA(n_comp)
    pca_model = pca.fit(X_scaled)
    X_pca = pca_model.transform(X_scaled) 
    # pca_model = pca.fit_transform(X_scaled)
    return pca, pd.DataFrame(X_pca), pca_model, pca_scaler 
    
# -------------------------------------------------------------------
# do pca imputing - pca preparation   
# -------------------------------------------------------------------
def do_pca_imputing(p_df, p_strategy): 
    """
    p_strategy: most_frequent (categorial) or mean
    """
    imp_mean = Imputer(missing_values=np.nan, strategy=p_strategy)
    imp_mean.fit(p_df)
    df_imputed = pd.DataFrame(imp_mean.transform(p_df), columns=p_df.columns)
    return df_imputed  

# -------------------------------------------------------------------
# do pca transformation with an existing model - feature reduction  
# -------------------------------------------------------------------
def do_pca_model( p_df, p_model): 
    """
    # do pca transformation with an existing model - feature reduction  
    """
    X_scaled = StandardScaler().fit_transform(p_df)
    X_pca =p_model.transform(X_scaled) 
    # pca_model = pca.fit_transform(X_scaled)
    return pd.DataFrame(X_pca), pd.DataFrame(X_scaled) 

# -------------------------------------------------------------------
# reencode some columns 
# replace characters by integer 
# -------------------------------------------------------------------
def do_reencode(p_df, p_me):
    """
    # reencode some columns 
    # replace characters by integer 
    """
    col_name = 'OST_WEST_KZ'
    if col_name in p_df.columns:
        col_old = p_df[col_name].unique() 
        p_df[col_name].replace(to_replace={'W': 0, 'O': 1}, inplace = True)
        if p_me == 'PRINT': 
            print(col_name)
            print('old', col_old) 
            print('new', p_df[col_name].unique()) 

    col_name = 'CAMEO_DEUG_2015' 
    if col_name in p_df.columns:
        col_old = p_df[col_name].unique() 
        p_df[col_name].replace(to_replace={'X': np.nan, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 
                                         '8': 8, '9': 9}, inplace = True)
        if p_me == 'PRINT': 
            print(col_name)
            print('old', col_old) 
            print('new', p_df[col_name].unique()) 
    return p_df 

# -------------------------------------------------------------------
# reencode 
# onehot encoding for categorial columns with many values 
# -------------------------------------------------------------------
def do_reencode_multi(p_df, p_cols_dic, p_me):
    """
    # reencode 
    # onehot encoding for categorial columns with many values 
    """
    df_cols_dic = pd.DataFrame.from_dict(p_cols_dic, orient='index')
    # columns of type categories with multiple values 
    df_cols_dic_multi = df_cols_dic[(df_cols_dic['data_type'] == 'categorical') & (df_cols_dic['skip'] == False )
                                            & (df_cols_dic['unique_no'] > 2 )]
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
    df_multi_imputed = do_pca_imputing(p_df[cols_multi], 'most_frequent') 
    # df_multi_dummies = pd.get_dummies(df_multi_imputed)
    if p_me == 'PRINT':     
        print('Before OneHotEncoded: ', p_df.shape)
    model_onehot = OneHotEncoder().fit(df_multi_imputed)
    # print('Model OneHotEncoded:', model_onehot) 
    # onehot encoding
    df_multi_onehot = pd.DataFrame((OneHotEncoder().fit_transform(df_multi_imputed)).toarray(), 
                                   columns=cols_label_multi)
    # remove the old columns 
    p_df.drop(cols_multi, axis = 1, inplace = True)
    p_df.reset_index(inplace = True, drop = True )
    # add the new encoded columns
    p_df = pd.concat([p_df, df_multi_onehot], axis = 1) 
    if p_me == 'PRINT':     
        print('After OneHotEncoded: ', p_df.shape) 

    return p_df, df_multi_onehot 


# -------------------------------------------------------------------
# reencode the PRAEGENDE_JUGENDJAHRE column 
# split in new columns for decade and movement
# -------------------------------------------------------------------
def do_reencode_pj_jugend(p_df, p_me):
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
        2, 2, 2, 2, 2, 2, 2
    ]
    p_df['PJ_MOVEMENT'] = np.select(condlist = cond_move, choicelist = value_move, default=1) 

    if p_me == 'PRINT':
        print(col_name)
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
def do_reencode_cameo(p_df, p_me):
    """
    # reencode the CAMEO_INTL_2015 column 
    # split in new columns for wealth and life staged
    """
    col_name = 'CAMEO_INTL_2015'
    if col_name not in p_df.columns:
        return p_df 

    # wealth 
    # cond_wealth = [
    #     (p_df[col_name] >= '11' ) & (p_df[col_name] <= '19' ), # wealthy
    #     (p_df[col_name] >= '21' ) & (p_df[col_name] <= '29' ), # prosperous
    #     (p_df[col_name] >= '31' ) & (p_df[col_name] <= '39' ), # comfortable
    #     (p_df[col_name] >= '41' ) & (p_df[col_name] <= '49' ), # less affluent
    #     (p_df[col_name] >= '51' ) & (p_df[col_name] <= '59' )  # poorer
    # ]
    # value_wealth = [
    #     1, 2, 3, 4, 5
    # ]
    # p_df['CAMEO_WEALTH'] = np.select(cond_wealth, value_wealth) 
    # # life stage 
    # cond_lifestage = [
    #     (p_df[col_name][:1] == 1 ), # maintream
    #     (p_df[col_name][:1] == 2 )  # maintream
    # ]
    # value_lifestage = [
    #     1, 2
    # ]
    # p_df = p_df.assign(CAMEO_WEALTH = lambda x: (p_df[col_name] // 10))
    # p_df = p_df.assign(CAMEO_LIFESTAGE = lambda x: (math.floor(p_df[col_name] / 10 ) ))
    # p_df = p_df.assign(CAMEO_LIFESTAGE = lambda x: (int(p_df[col_name]) % 10))
    p_df[col_name].replace(to_replace={'XX': 99}, inplace = True)
    p_df[col_name].replace(to_replace={np.nan: 99}, inplace = True)
    # p_df['CAMEO_LIFESTAGE'] = p_df[col_name] // 10 
    p_df['CAMEO_LIFESTAGE'] = p_df[col_name].astype(int) // 10 
    p_df['CAMEO_LIFESTAGE'].replace(to_replace={9: np.nan}, inplace = True)
    p_df['CAMEO_WEALTH'] = p_df[col_name].astype(int) % 10 
    p_df['CAMEO_WEALTH'].replace(to_replace={9: np.nan}, inplace = True)
    if p_me == 'PRINT':
        print(col_name)
        print('old', p_df[col_name].unique())
        print('new wealth', p_df['CAMEO_WEALTH'].unique()) 
        print('new lifestage', p_df['CAMEO_LIFESTAGE'].unique()) 

    # drop the original column 
    p_df.drop(axis = 1, columns = [col_name], inplace = True)  
    return p_df 

# -------------------------------------------------------------------
# helper to update the column dictionary
# -------------------------------------------------------------------
def upd_cols_skip(p_col_name, p_reason, p_cols_dic ):
    """
    # helper to update the column dictionary
    """
    p_cols_dic[p_col_name]['missing_reason'] = p_reason  
    p_cols_dic[p_col_name]['skip'] = True 
    return p_cols_dic 

# -------------------------------------------------------------------
# mark certain columns which should be removed in the column 
# dictionary as to remove (SKIP) together with a reason code
# -------------------------------------------------------------------
def add_cols_2_remove( p_cols_dic  ):
    """
    # mark certain columns which should be removed in the column 
    # dictionary as to remove (SKIP) together with a reason code
    """
    p_cols_dic = upd_cols_skip('GEBAEUDETYP', 'diff.company/resedential not relevant', p_cols_dic)
    p_cols_dic = upd_cols_skip('KONSUMNAEHE', 'distance to RA1 cell not relevant', p_cols_dic)
    p_cols_dic = upd_cols_skip('CAMEO_DEU_2015', 'details for CAMEO_DEUG_2015', p_cols_dic)
    p_cols_dic = upd_cols_skip('PLZ8_ANTG1', 'represented by PLZ8_BAUMAX', p_cols_dic)
    p_cols_dic = upd_cols_skip('PLZ8_ANTG2', 'represented by PLZ8_BAUMAX', p_cols_dic)
    p_cols_dic = upd_cols_skip('PLZ8_ANTG3', 'represented by PLZ8_BAUMAX', p_cols_dic)
    p_cols_dic = upd_cols_skip('PLZ8_ANTG4', 'represented by PLZ8_BAUMAX', p_cols_dic)
    p_cols_dic = upd_cols_skip('LP_FAMILIE_FEIN', 'details for LP_FAMILIE_GROB', p_cols_dic)
    p_cols_dic = upd_cols_skip('LP_STATUS_FEIN', 'details for LP_STATUS_GROB', p_cols_dic)
    return p_cols_dic 

# -------------------------------------------------------------------
# get all the columns out of the feature list and extract some info
# about the missing values and the column data type
# -------------------------------------------------------------------
def get_cols_dic( p_df, p_feat, p_nan_rate ):
    """
    # get all the columns out of the feature list and extract some info
    # about the missing values and the column data type
    """
    total_rec_cnt = p_df.shape[0]
    cols_type = p_df.dtypes
    cols_dic = {}
    for col_name in p_df.columns:
        col_type = cols_type[col_name]
        col_int = True if col_type == 'int64' or col_type == 'float64' else False 
        missing_int = []
        missing_list = []
        # what is the representation of the unknown values  
        missing_str = p_feat.loc[p_feat['attribute'] == col_name]['missing_or_unknown'] 
        col_datatype = p_feat.loc[p_feat['attribute'] == col_name]['type'].values[0]
        if len(missing_str.values) > 0: 
            # convert string to list 
            missing_list = list(missing_str.values[0].replace('[', '').replace(']', '').split(','))
            # convert search string to integer if necessary
            if col_int == True:  
                for item in missing_list:
                    if item == 'X' or item == 'XX': item = '-1'
                    if item == 'XX': item = '-1'
                    # print('col', col_name, col_int )
                    if item != '':
                        missing_int.append(int(item))
                missing_list = missing_int 
                missing_list.append(np.nan)
            else:
                missing_list.append(np.nan)
        # info about the missed values 
        missing_cnt = len(p_df.loc[p_df[col_name].isin(missing_list)]) 
        missing_rate =  missing_cnt / p_df.shape[0] 
        if missing_rate > p_nan_rate:
             missing_skip = True
             missing_reason = 'high rate of NaNs'
        else:
            missing_skip = False
            missing_reason = ''
        unique_no = p_df[col_name].nunique()
        cols_dic[col_name] = { 'col_name': col_name, 'col_type': col_type, 'missing_list': missing_list, 
                               'data_type': col_datatype, 'skip': missing_skip, 'missing_cnt': missing_cnt, 
                               'missing_rate': missing_rate, 'missing_reason': missing_reason, 
                               'unique_no': unique_no  } 
    return cols_dic 

# -------------------------------------------------------------------
# get all the columns with a certain rate of missing values and 
# mark them in the column dictionary as to remove ("SKIP")
# -------------------------------------------------------------------
def get_cols_nan( p_df, p_cols_dic, p_nan_rate, p_set_skip ):      
    """
    # get all the columns with a certain rate of missing values and 
    # mark them in the column dictionary as to remove ("SKIP")
    """
    for col_dic in p_cols_dic.values():
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
            p_cols_dic[col_name]['skip'] = missing_skip
            p_cols_dic[col_name]['missing_reason'] = missing_reason 
                
        p_cols_dic[col_name]['missing_cnt'] = missing_cnt
        p_cols_dic[col_name]['missing_rate'] = missing_rate 
        # print('missing ', col_name, 'counter: ', missing_cnt, missing_rate )  
    return p_cols_dic 

# -------------------------------------------------------------------
# replace all the values which are marked as missing values in the 
# column dictionary with np.nan
# -------------------------------------------------------------------
def replace_cols_nan( p_df, p_cols_dic):
    """
    # replace all the values which are marked as missing values in the 
    # column dictionary with np.nan
    """
    for col_dic in p_cols_dic.values():
        missing_list = col_dic['missing_list']
        col_name = col_dic['col_name']
        nan_list = []
        for missing_item in missing_list:
            nan_list.append(np.nan)
        p_df[col_name].replace(to_replace = missing_list, value = nan_list, inplace = True)
    return p_df 

# -------------------------------------------------------------------
# delete the columns of the dataframe which are marked as SKIP in the 
# columns dictionary 
# -------------------------------------------------------------------
def del_cols_nan( p_df, p_cols_dic, p_me ):
    """
    # delete the columns of the dataframe which are marked as SKIP in the 
    # columns dictionary 
    """
    remove_cols = []
    for col_dic in p_cols_dic.values():
        col_name = col_dic['col_name']
        # remove column if marked as to skip 
        if col_dic['skip'] == True:
            if p_me == 'PRINT':
                print('remove', col_name)
            remove_cols.append(col_name)
    if len(remove_cols) > 0:
       p_df_res = p_df.drop(axis = 1, columns = remove_cols )
    if p_me == 'PRINT':
        print('DataFrame after Column Deletion:', p_df_res.shape)
    return p_df_res

# -------------------------------------------------------------------
# extract the master data from the md-file 
# -------------------------------------------------------------------
def get_master_data( p_path_in ): 
    """
    # extract the master data from the md-file 
    """
    attr_dic = {}
    skip = True  

    for p in Path(p_path_in).glob('*.md'):
        # print(p.name)
        fi = open(os.path.join(p_path_in, p.name), 'r')
#       fo_name = p.name.replace('.srt', '.txt')
#       fo = open(os.path.join(path_out, fo_name), 'w') 
        lines = fi.readlines()

        for line in lines: 
            line.strip() 
            line = line.replace("\n", "")
        #   start with line ### 1.1. AGER_TYP
            if ( '1.1. AGER_TYP' in line): 
                skip = False
            if (skip == True): 
                continue 
            if line[0:3] == '###':
                # print('split', line.rsplit(' '))
                attr = line.rsplit(' ')[2]
                # print('split', attr) 

            if line[0:1] == '-': 
                attr_list = line[1:].rsplit(':')
                if (len(attr_list) > 1):
                    attr_key = attr + '|' + attr_list[0].lstrip()
                    attr_value = line[1:].rsplit(':')[1].lstrip()
                    attr_dic[attr_key] = attr_value
    return attr_dic

# -------------------------------------------------------------------
# print some statistic data depending on the chosen method P_ME 
# -------------------------------------------------------------------
def print_stat(p_me, p_df):
    """
    # print some statistic data depending on the chosen method P_ME 
    """
    set_print_options('ALL')
    if p_me == 'NO' or p_me == 'ALL':
        print('--------------------------------------------------')
        print('Numbers NO')
        print('--------------------------------------------------')
        print('Number of rows {:,} and columns {:,}'.format(p_df.shape[0], p_df.shape[1]))
        print('Number of NaN values in our DataFrame:', p_df.isnull().sum().sum())
        print('Number of columns with NaN values:', p_df.isnull().any().sum())
        print('Number of rows with NaN columns > {:,}:'.format(gc_max_nan_columns), 
            (p_df.loc[:, :].isnull().T.sum().values > gc_max_nan_columns).sum())
    
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

    set_print_options()

# -------------------------------------------------------------------
# print the feature data together with info from the column dictionary 
# -------------------------------------------------------------------
def print_feat(p_df, p_feat, p_cols_dic, p_sel_datatype):
    """
    # print the feature data together with info from the column dictionary 
    """
    set_print_options('ALL')
    print('--------------------------------------------------')
    print('Column Features')
    print('--------------------------------------------------\n')
    #print(p_cols_dic)
    # print(p_feat.sort_values(by='attribute')) 

    for feat in p_feat.sort_values(by='attribute').values: 
        col_name = feat[0]
        cnt = len(col_name)
        col_type = p_cols_dic[col_name]['col_type']
        data_type = p_cols_dic[col_name]['data_type']
        if len(p_sel_datatype) > 0: 
            if data_type not in p_sel_datatype: 
                continue
        col_skip = 'Skip' if p_cols_dic[col_name]['skip'] == True else ''
        tabs = ( '\t' if len(col_name) > 13 else '\t\t' )
        if col_name in p_df.columns:
            print('{:5s} {:25s} {:10s} {:12s} {:60s}'.format(str(col_skip), col_name, str(col_type), data_type, 
                            str(p_df[col_name].unique())[0:60] ))
    set_print_options()

# -------------------------------------------------------------------
# print the feature data and additonal info about the no. of 
# rows per type  
# -------------------------------------------------------------------
def print_feat_info():
    """
    # print the feature data and additonal info about the no. of 
    # rows per type  
    """
    set_print_options('ALL')
    feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', delimiter=';')
    print('--------------------------------------------------')
    print('Column Features Info')
    print('--------------------------------------------------\n')
    print('Feature Number: ', feat_info['type'].count())
    print(feat_info.groupby(by='type').size())
    print(feat_info)
    set_print_options()

# -------------------------------------------------------------------
# print the column dictionary 
# -------------------------------------------------------------------
def print_colsdic(p_cols_dic):
    """
    # print the column dictionary 
    """
    set_print_options('ALL')
    cols_df = pd.DataFrame.from_dict(p_cols_dic, orient='index')
    cols_df = cols_df.sort_values(by=["col_name"])
    print('--------------------------------------------------')
    print('Column Dictionary')
    print('--------------------------------------------------\n')
    print(str(cols_df.head(100)))
    set_print_options()

def print_colsdic2(p_cols_dic):
    cols_df = pd.DataFrame.from_dict(p_cols_dic, orient='index')
    cols_df.iloc[:,1:]
    print(str(cols_df.iloc[:,1:]))

# -------------------------------------------------------------------
# print general/customer compare  
# -------------------------------------------------------------------
def print_colcompare(p_df_gen, p_df_cust):
    """
    # print general/customer compare  
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print('Gen/Customer', p_df_gen.shape, p_df_cust.shape) 
    for col in p_df_gen.columns: 
        if col not in  p_df_cust.columns:
            print('General Column without Customer Column', col)
    for col in p_df_cust.columns:
        if col not in p_df_gen.columns:
            print('Customer Column without General Column', col)
            
# -------------------------------------------------------------------
# print the pca values for all principal components
# -------------------------------------------------------------------
def print_pca(p_pca, p_comp_no, p_df):
    """
    # print the pca values for all principal components
    """
    feat_names_init = p_df.columns 

    # # most important feature for every pc
    most_important = [p_pca.components_[ii].argmax() for ii in range(p_comp_no)]

    # get the names
    feat_names_important = [feat_names_init[most_important[ii]] for ii in range(p_comp_no)]
    feat_values_important = [np.abs(p_pca.components_[ii]).max() for ii in range(p_comp_no)]
    feat_important = list(zip(feat_names_important, feat_values_important)) 

    dic2 = {'PC{}'.format(ii+1): feat_important[ii]  for ii in range(p_comp_no)}
    # build the dataframe
    df = pd.DataFrame(list(dic2.items()),columns=['PCA', 'Feature, Weight'] ) 
     
    print('Most important feature per Component by Weights\n', df.head( p_comp_no))   

# -------------------------------------------------------------------
# print the pca weights for 1 chosen principal component sorted 
# by the weights value 
# -------------------------------------------------------------------
def print_pca_one(p_pca, p_comp_no, p_df): 
    """
    # print the pca weights for 1 chosen principal component sorted 
    # by the weights value 
    """
    feat_names_init = p_df.columns 
    feat_values_init = p_pca.components_[p_comp_no]
    feat = list(zip(feat_names_init, feat_values_init))
    feat_size = len(feat) 
    feat.sort(key = lambda i: i[1], reverse=True)
    print('Principal Component {} by Weigths\n'.format(p_comp_no))
    [print(feat[i][0], "%0.5f" % feat[i][1]) for i in range(feat_size)]
    df_feat = pd.DataFrame(feat, columns=['Feature', 'Weight']) 
    print(df_feat.head()) 

# -------------------------------------------------------------------
# print the feature with the top weights for every principal component
# of the PCA model  
# -------------------------------------------------------------------
def print_pca_most(p_pca, p_comp_no, p_feat):
    """
    # print the feature with the top weights for every principal component
    # of the PCA model  
    """
    for i in range(p_comp_no):
        print('index', i)
    exit() 
    # get the index of the most important feature on EACH component i.e. largest absolute value
    # using LIST COMPREHENSION HERE
    most_important = [np.abs(p_pca.components_[i]).argmax() for i in range(p_comp_no)]
    print('most', most_important )
    feat_names_init = p_feat['attribute'].values
    # get the names
    feat_names_important = [feat_names_init[most_important[i]] for i in range(p_comp_no)]
    feat_values_important = [np.abs(p_pca.components_[i]).max() for i in range(p_comp_no)]
    feat_important = list(zip(feat_names_important, feat_values_important))
    [print('PC{}'.format(i+1), feat_important[i][0], "%0.2f" % feat_important[i][1]) for i in range(p_comp_no )]  
    # [print( name, value) for name, value in zip(feat_names_important, feat_values_important)]
    # using LIST COMPREHENSION HERE AGAIN
    # dic = {'PC{}'.format(i+1): feat_names_important[i]  for i in range(p_comp_no)}
    # print(type(feat_important))
    # dic2 = {'PC{}'.format(i+1): feat_important[i]  for i in range(p_comp_no)}
    # print(list(dic2))
    # build the dataframe
    # df = pd.DataFrame(sorted(dic.items()))   
    #print(df.head())   

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
# plot a box plot of columns   
# -------------------------------------------------------------------
def plot_cols(p_df, p_cols, p_master_data): 
    """
    # plot a box plot of columns   
    """
    plot_ind = 1
    for col in p_cols:
        col_name = col 
        plt.figure(figsize=(10,5))
        # plot a bar chart for the column
        ax = sns.countplot(data=p_df, x = col_name, color = gc_base_color)
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
            ax = sns.countplot(data=p_df, x = col_name, color = gc_base_color)
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
        
# -------------------------------------------------------------------
# plot the kmeans scores - elbough plot   
# -------------------------------------------------------------------
def plot_kmeans_cluster(p_cluster, p_cluster2):
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
    ax.bar(ind,vals, color = 'b', width=width, edgecolor = 'black', label='General')
    if len(p_cluster2) > 0:
        vals2 = p_cluster2[:, 1]  
        ax.bar(ind + width + 0.05,vals2, color = 'g', width = width, edgecolor = 'black', label='Customer')
    ax.plot(ind)
    for ii in range(num_x):
        val = "{:+.2f}".format(vals[ii])
        ax.annotate(val, xy=(ind[ii], vals[ii]), va="bottom", ha="center", fontsize=8)
        if len(p_cluster2) > 0: 
            val2 = "{:+.2f}".format(vals2[ii])
            ax.annotate(val2, xy=(ind[ii]+width + 0.05, vals2[ii]), va="bottom", ha="center", fontsize=8)

    # ax.xaxis.set_tick_params(width=0)
    # ax.yaxis.set_tick_params(width=1, length=12)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customers")
    plt.xticks(ind + width/2,ind) 
    plt.title('Elements per Cluster') 
    plt.legend()
    plt.show()

# -------------------------------------------------------------------
# plot the kmeans scores - elbough plot   
# -------------------------------------------------------------------
def plot_kmeans_scores(p_cluster_nos, p_scores):
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

# -------------------------------------------------------------------
# plot the elbow curve for the PCA model  
# --> "scree_plot" function of the course 
# -------------------------------------------------------------------
def plot_pca(pca):
    """
    # plot the elbow curve for the PCA model  
    # --> "scree_plot" function of the course 
    """
    num_comp = len(pca.explained_variance_ratio_ )
    ind = np.arange(num_comp)
    vals = pca.explained_variance_ratio_
    cumvals = np.cumsum(vals)
    df_cumvals = pd.concat([pd.DataFrame(ind), pd.DataFrame(vals), pd.DataFrame(cumvals)], axis = 1) 
    df_cumvals.columns = [ 'Component', 'Variance', 'Cumul.Variance']
    
    plt.figure(figsize=(16,8 ))
    ax = plt.subplot(111)
   
    ax.bar(ind,vals)
    ax.plot(ind,cumvals)
    for i in range(num_comp):
        ax.annotate(r"%s%%" % ((str(vals[i]*100))[:4]), (ind[i]+0.2, vals[i]), 
                    va="bottom", ha="center", fontsize=12)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Pricipal Component")
    ax.set_ylabel("Variance Explainded (%)")
    plt.title('Explained Variance Per Principal Component') 
    plt.show()
    return df_cumvals 

# -------------------------------------------------------------------
# plot the PCA components 
# -------------------------------------------------------------------
def plot_pca_component(pca, p_comp, p_df):
    """
    # plot the PCA components 
    """
    df_plot = pd.DataFrame(list(zip(p_df.columns, pca.components_[p_comp])), columns=['col_name', 'weight' ])
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
        ax.annotate(val, xy=(ind[ind_res], vals[ind_res]), va="bottom", ha="center", fontsize=8)
    cumvals = np.cumsum(vals)
    ax.bar(ind,vals)
    # ax.plot(ind,cumvals)
    ax.set_xticks(ind)
    ax.set_xlabel("Features")
    ax.set_ylabel("Weights")
    # plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.title('Top {} Features, Bottom {} Features for Component {}'.format(10, 10, p_comp)) 
    plt.show()

# -------------------------------------------------------------------
# plot the PCA components -> s. plot_pca  
# -------------------------------------------------------------------
def plot_pca_component2(pca):
    """
    # plot the PCA components -> s. plot_pca  
    """
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='blue')
    plt.xlabel('PCA Components')
    plt.ylabel('Variance %') 
    plt.xticks(features)
    plt.show() 


# -------------------------------------------------------------------
# plot of the distribution of rows with a lot of missing values 
# compared to the rows with valid values   
# -------------------------------------------------------------------
def plot_row_nan(p_col_name, p_df_nan, p_df_filled):
    """
    # plot of the distribution of rows with a lot of missing values 
    # compared to the rows with valid values   
    # plt.figure(figsize=(10,6 ))
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(12,7))
    plt.subplot(1, 2, 1)
    plt.title("Lots NaN Values")
    axs[0] = sns.countplot(data=p_df_nan, x = p_col_name, color = gc_base_color)
    plt.subplot(1, 2, 2)
    plt.title("Few NaN Values")
    axs[1] = sns.countplot(data=p_df_filled, x = p_col_name, color = gc_base_color)

# -------------------------------------------------------------------
# plot of the distribution of rows with a lot of missing values 
# compared to the rows with valid values   
# -------------------------------------------------------------------
def plot_rows_nan(p_cols_dic, p_df_nan, p_df_filled, p_max_img):
    """
    # plot of the distribution of rows with a lot of missing values 
    # compared to the rows with valid values   
    # plt.figure(figsize=(10,6 ))
    """
    # plt.figure(figsize=(10,6 ))
    ii = 0
    for key, value in p_cols_dic.items():
        if value['skip'] == False:
            ii+=1
            plot_row_nan(key, p_df_nan, p_df_filled) 
        if ii>p_max_img: break  
    plt.show() 

