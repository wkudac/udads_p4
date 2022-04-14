import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import math 
from datetime import datetime
from pytz import timezone

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.pipeline import Pipeline 
from sklearn.utils import resample

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

# from imblearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay

GC_UTC = timezone('UTC')
GC_BASE_COLOR = sns.color_palette()[0]
GC_COLORMAP = plt.cm.RdBu

class SLSegment: 
    """ 
    class with some helper methods to print statistics around a dataset 
    """
    
    def __init__(self, p_df_label, p_df_feat, p_unbalanced=False, 
                p_print=True, p_plot=True):
        """
        Instantiates the object with the dataset 
        Input: 
        - Dataset to analyze
        """
        self.set_print(p_print)
        self.set_plot(p_plot) 
        self.X_feat = p_df_feat 
        self.y_label = p_df_label 
        self.unbalanced = p_unbalanced 

        # if p_unbalanced == True: 
        #     self.data_split_unbalanced( p_df_label, p_df_feat)
        # else:    
        #     self.data_split( p_df_label, p_df_feat)

    def set_print(self, p_print): 
        self.me_print = p_print 

    def set_plot(self, p_plot): 
        self.me_plot = p_plot 

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

    def data_split(self): 
        """
        Split the data set into test and train data 
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split( self.X_feat, self.y_label, test_size = .2, random_state = 42 )
        
        if self.me_print == True:
          print('x_train / x_test: ', self.X_train.shape, self.X_test.shape, 
                    self.X_test.shape[0] / self.X_train.shape[0] )
          print('y_train / y_test: ', self.y_train.shape, self.y_test.shape, 
                    self.y_test.shape[0] / self.y_train.shape[0] )

    def data_split_unbalanced(self): 
        """
        Split the data set into test and train data if the data set contains imbalanced data 
        Use Oversampling to generate additional data for the minority class 
        """
        # split before the train data will be oversampled to 
        # keeep a clean test data set 
        self.data_split( )
        print('before', len(self.y_label), len(self.X_feat) )
        print('before', len(self.y_train), len(self.X_train) )

        # Create oversampled training data set for minority class
        X_oversampled, y_oversampled = resample(self.X_train[self.y_train == 1],
                                    self.y_train[self.y_train == 1],
                                    replace=True,
                                    n_samples= self.X_train[self.y_train == 0].shape[0],
                                    random_state=42)
        # Append the oversampled minority class to training data and related labels
        self.X_train = np.vstack((self.X_train[ self.y_train == 0], X_oversampled))
        self.y_train = np.hstack(( self.y_train[ self.y_train == 0], y_oversampled))
        print('after', len(self.y_label), len(self.X_feat) )
        print('after', len(self.y_train), len(self.X_train) )

    # def data_split(self, p_df_label, p_df_feat): 
    #     """
    #     Split the data set into test and train data 
    #     """
    #     self.X_train, self.X_test, self.y_train, self.y_test = \
    #             train_test_split( p_df_feat, p_df_label, test_size = .2, random_state = 42 )
        
    #     if self.me_print == True:
    #       print('x_train / x_test: ', self.X_train.shape, self.X_test.shape, 
    #                 self.X_test.shape[0] / self.X_train.shape[0] )
    #       print('y_train / y_test: ', self.y_train.shape, self.y_test.shape, 
    #                 self.y_test.shape[0] / self.y_train.shape[0] )

    # def data_split_unbalanced(self, p_df_label, p_df_feat): 
    #     """
    #     Split the data set into test and train data if the data set contains imbalanced data 
    #     Use Oversampling to generate additional data for the minority class 
    #     """
    #     # split before the train data will be oversampled to 
    #     # keeep a clean test data set 
    #     self.data_split(p_df_label, p_df_feat)
    #     print('before', len(p_df_label), len(p_df_feat) )

    #     # Create oversampled training data set for minority class
    #     X_oversampled, y_oversampled = resample(self.X_train[self.y_train == 1],
    #                                 self.y_train[self.y_train == 1],
    #                                 replace=True,
    #                                 n_samples= self.X_train[self.y_train == 0].shape[0],
    #                                 random_state=42)
    #     # Append the oversampled minority class to training data and related labels
    #     self.X_train = np.vstack((self.X_train[ self.y_train == 0], X_oversampled))
    #     self.y_train = np.hstack(( self.y_train[ self.y_train == 0], y_oversampled))
    #     print('after', len(self.y_train), len(self.X_train) )


        # # Create oversampled training data set for minority class
        # X_oversampled, y_oversampled = resample(p_df_feat[p_df_label == 1],
        #                             p_df_label[p_df_label == 1],
        #                             replace=True,
        #                             n_samples=p_df_feat[p_df_label == 0].shape[0],
        #                             random_state=42)
        # # Append the oversampled minority class to training data and related labels
        # X_balanced = np.vstack((p_df_feat[p_df_label == 0], X_oversampled))
        # y_balanced = np.hstack((p_df_label[p_df_label == 0], y_oversampled))

        # self.data_split(y_balanced , X_balanced)

  
    def do_build(self, p_estimators=['LOGREG', 'RANFOR', 'XGB'], p_para={} ): 
        """"
        All the steps from the creation of the model object, the training of the model,  
        the prediction for test data and printing the results of model evaluation  
        Input:
        - p_estimators: models for which the prediction should be done
        """
        # train classifier 
        ests = p_estimators
        model_res = {}
        for est in ests:  # [1:2]:
            
            # build the model 
            self.print_progress('FREE', est + ' build') 
            model, title = self.model_build(est, p_para)
            
            # train the model
            self.print_progress('FREE', est + ' fit') 
            model.fit(self.X_train, self.y_train)
            
            # predict on test data 
            self.print_progress('FREE', est + ' predict') 
            y_pred = model.predict(self.X_test)
            
            # print the evaluation result 
            self.print_progress('FREE', est + ' result') 
            res = self.model_result(model, title, self.y_test, y_pred)
            model_res[est] = res 
            # model.get_params().keys() 
            df_model_res = pd.DataFrame.from_dict(model_res, orient='index')
        return df_model_res 
      
    def model_result(self, p_model, p_title, y_test, y_pred):
        """
        Print the model results mainly based on the confusion matrix and the ROC/auc scores
        """
        print('\n')
        self.print_progress('FREE', p_title) 
        labels = np.unique(y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
        if self.me_print == True:
            print("Labels:", labels)
            print("Confusion Matrix: \n", confusion_mat)
 
        # if self.me_plot == True:
        #     cm = confusion_matrix(y_test, y_pred, labels=p_model.classes_)
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm,
        #                                 display_labels=p_model.classes_)
        #     fig, ax = plt.subplots(figsize=(8, 6))
        #     ax.set_title(p_title, fontsize=18)
        #     ax.set_xlabel(xlabel=p_model.classes_, fontsize=18)
        #     ax.set_ylabel(ylabel=p_model.classes_, fontsize=18)
        #     disp.plot(ax=ax)
        #     plt.show()

        if self.me_plot == True: 
            self.model_res_plot_coma(p_title, p_model, y_test, y_pred)

        if self.me_plot == True: 
            self.model_res_plot_roc(p_title, y_test, y_pred)

        accuracy = (y_pred == y_test).mean() 
        # ra_score = roc_auc_score( y_test, y_pred)
        if self.me_print == True:
            print("Accuracy        :", accuracy )
            print('Mean Score      :', p_model.cv_results_['mean_test_score'].mean())
            print('Mean Fit Time   :', p_model.cv_results_['mean_fit_time'].mean() )
           # precision_recall_fscore_support(y_test, y_pred, labels=['default', 'paid off'])
            print('ROC AUC accuracy: {:.2f}:'.format(roc_auc_score( y_test, y_pred)))
            print('Best Parameters :\n', p_model.best_params_)
        
        res = { 'accuracy': accuracy, 
                'auccuracy_roc_auc': roc_auc_score( y_test, y_pred),
                'mean_score': p_model.cv_results_['mean_test_score'].mean(),
                'mean_fit_time': p_model.cv_results_['mean_fit_time'].mean(),
                'model': p_model 
            }
        # print('result', p_model.cv_results_)
        return res 
    
    def model_build(self, p_me='LOGREG', p_para={}):
        """
        Create the model object for a certain estimator, use KFold and RandomizedSearchCV 
        to prepare the object for repetitive train/test splittings and the search 
        for the best hyper parameters 
        input:
        - p_me: model which should be build
        - p_para: dictionary with hyper parameters for the model 
        """
        # external parameter settings
        if p_me in p_para: 
            model_parm = p_para[p_me]
        else:
            model_parm = {}

        if p_me == 'RANFOR':
            clf, para, title = self.model_build_randomforest( model_parm)
        elif p_me == 'BALRANFOR':
            clf, para, title = self.model_build_balanced_randomforest( model_parm)
        elif p_me == 'XGB':
            clf, para, title = self.model_build_xgb( model_parm)
        elif p_me == 'LOGREG':
            clf, para, title = self.model_build_logisticregression( model_parm)

        # over = RandomOverSampler(sampling_strategy=0.09)  # 0.333
        # under = RandomUnderSampler(sampling_strategy=0.888) # 0.666
        # steps = [('o', over), ('u', under), ('clf', clf)]
        # pipeline = Pipeline(steps=steps)

        pipeline = Pipeline( [
            # ('o', over),
            # ('u', under),
            ('clf', clf) 
        ])

        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
        rs = RandomizedSearchCV(pipeline, param_distributions=para, scoring='roc_auc', 
                            return_train_score=True, cv=cv, verbose=0)
        return rs, title 

    def model_build_randomforest( self, p_para={}): 
        """
        Create the RandomForest model object together with some parameters which should be used 
        in the RandomizedSearchCV to find the optimal constellation
        """
        title = 'RandomForest'
        clf = RandomForestClassifier()
         # data set is unbalanced - use oversampling during splitting
        if self.unbalanced == True: 
            self.data_split_unbalanced( )
        else:    
            self.data_split( )
        # default parameter
        if len(p_para) == 0:
            para = {
                'clf__max_features': list(range(1, self.X_test.shape[1]+1)),
                'clf__n_estimators': list(range(10,70)),
                # 'clf__min_samples_leaf': list(range(1,11)),
                # 'clf__class_weight': ["balanced"], 
                'clf__min_samples_split': list(range(2,11)),
                'clf__criterion': ["gini", "entropy"]
            }
        else:
            para = p_para 
            
        return clf, para, title 

    def model_build_balanced_randomforest(self, p_para={}): 
        """
        Create RandomForest model object for imbalanced data together with 
        some parameters which should be used in the RandomizedSearchCV 
        to find the optimal constellation
        """
        title = 'BalancedRandomForest'
        clf = BalancedRandomForestClassifier(random_state=42)
         # handling of unbalance data is done by method 
        self.data_split( )

        # default parameter
        if len(p_para) == 0:
            para = {
                'clf__max_features': list(range(1, self.X_test.shape[1]+1)),
                'clf__n_estimators': list(range(10,70,10)),
                'clf__sampling_strategy': [0.2,0.3,0.4],
                # 'clf__min_samples_leaf': list(range(1,11)),
                # 'clf__class_weight': ["balanced"], 
                'clf__min_samples_split': list(range(2,11)),
                'clf__criterion': ["gini", "entropy"]
            }
        else:
            para = p_para
        return clf, para, title 


    def model_build_xgb( self, p_para={}): 
        """
        Create the xgboost model object together with some parameters which should be used 
        in the RandomizedSearchCV to find the optimal constellation
        """

        title = 'XGB'
        clf = xgb.XGBClassifier()
         # data set is unbalanced - use oversampling during splitting
        if self.unbalanced == True: 
            self.data_split_unbalanced( )
        else:    
            self.data_split( )

        # default parameter
        if len(p_para) == 0:
            para = {
                'clf__objective': ['binary:logistic'],
                'clf__learning_rate': [0.1, 0.3, 0.9],
                'clf__n_estimators': [250],
                'clf__max_depth': [3,6,9],
                'clf__eval_metric': ['rmse']
            }
        else:
            para = p_para 
        return clf, para, title 

    def model_build_logisticregression( self, p_para={}): 
        """
        Create the logistic regression model object together with some parameters which should be used 
        in the RandomizedSearchCV to find the optimal constellation
        """

        title = 'Logistic Regression'
        clf = LogisticRegression()
         # data set is unbalanced - use oversampling during splitting
        if self.unbalanced == True: 
            self.data_split_unbalanced( )
        else:    
            self.data_split( )

        # default parameter
        if len(p_para) == 0:
            para = {
                'clf__C' : [0.1, 1, 10, 100, 1000],
                'clf__solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
                'clf__penalty': ['l1', 'l2', 'none'],
                'clf__max_iter' : [100, 1000,2500, 5000]
            }
        else: 
            para = p_para 

        return clf, para, title 

    def model_res_plot01(self, p_df):
        """
        Plot a bar chart for the model scores
        """

        plt.figure().clear() 
        plt.figure(figsize = (12,7))
        plt.title('Model Evaluation') 
        ax1 = p_df.iloc[:,-1].plot.bar()
        plt.show()

    def model_res_plot02(self, p_df):
        """
        Plot a bar chart for the model scores and some column annotations 
        """
        # num_comp = len(pca.explained_variance_ratio_ )
        df_plot = p_df.copy() 
        df_plot.fillna(0,inplace=True)
        ind = -1
        for col in df_plot.columns[0:-1]:
            ind += 1
            x_val = list(df_plot.index)
            y_val = list(df_plot.iloc[:,ind])
            plt.figure(figsize=(8,6 ))
            ax = plt.subplot(111)
            ax.bar(x_val,y_val)
            for i in range(len(df_plot)):
                val = "{:+.2f}".format(y_val[i])
                ax.annotate(val, xy=(x_val[i], y_val[i]), va="bottom", ha="center", fontsize=12)
            ax.set_ylabel('Score')
            plt.title(col) 
        plt.show()

    def model_res_plot_coma(self, p_title, p_model, p_y_test, p_y_pred): 
        cm = confusion_matrix(p_y_test, p_y_pred, labels=p_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=p_model.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(p_title, fontsize=18)
        ax.set_xlabel(xlabel=p_model.classes_, fontsize=18)
        ax.set_ylabel(ylabel=p_model.classes_, fontsize=18)
        disp.plot(ax=ax)
        plt.title('Confusion Matrx ' + p_title)
        plt.show()

    def model_res_plot_roc(self, p_title, p_y_test, p_y_pred): 
        fpr, tpr, thresholds = roc_curve(p_y_test, p_y_pred ) 
        auc = roc_auc_score(p_y_test, p_y_pred)
        #create ROC curve
        plt.plot(fpr,tpr,label="AUC="+str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.title('ROC curve ' + p_title)
        plt.show()