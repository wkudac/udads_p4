# Arvato Customer Segmentation 

## Introduction 

The main topic of this project is customer segmentation and prediction of potential new customers which is still an ongoing task in the business world. The initiative comes from a company which tries to improve their email campaigns by sending out emails only to recipients who would be proper customers. That saves money and doesn’t disturb customers who do not want to receive emails by foreign companies in topics they are not interested in. 
The customer segmentation is done by Unsupervised Machine Learning techniques using PCA (Principal Component Analysis ) and KMeans Clustering while  Supervised Machine Learning models (Logistic Regression, RandomForest and AgBoost) are used to make some predictions about potential customers. 


## Steps 

1.	In the 1st part some preprocessing is done to prepare the data for further Machine Learning techniques. 
2.	In the 2nd part PCA (Principal Component Analysis) and KMeans is used to get some clusters in which the target population can be grouped. 
3.	The 3rd part goes one step further and uses the output of the 2nd step to apply some Supervised Machine Learning models to make some predictions about new customers.

## Project Motivation

The project was done as part of a Udacity DataScience course. The interesting thing for me was the combination of Unsupervised and Supervised Machine Learning techniques in one continuous and connected approach.  For me as a beginner it was always impressive how much work it is to prepare the data and how different techniques can be combined. 

The principals components and the clusters created by Unsupervised techniques like PCA and KMeans Clustering can be further used in different Supervised Machine Learning models like Logistic Regression, Random Forest and AgBoost to make some predictions. 

Basically the approach in such a project is often similar. Have a look at the data, extract and preprocess only the data which might be useful for you and apply some modern Machine Learning techniques on them. 

There are powerful modules provided in the community like scikit-learn which alleviates the work very much. But there is still work to do outside of that modules especially in the data preparation part.   
I have tried to organize the code a little bit to make it more reusable and during the course it becomes clearer what are the tasks which appear repeatedly. But to be honest there is much room for improvements.  An interesting thing is that especially in such a topic like AI there is much manual work to do. 

A more detailed description with further results can be found in this [blog post](https://medium.com/@wkudac/stackoverflow-survey-2017-where-are-you-f802d450fa1).
 

## File Descriptions 

The data basis consist of 6 files which are provided by a company. The data is not attached to the Github repository because it contains confidential data about customers so that you will find here only the Jupyter notebook and the helper Python code to process the data. 
The data files are all located in a subfolder called “data”. 
1.	DIAS Attributes – Values 2017.xlsx<br>
Contains possible attributes for a lot of the columns in the data set which can be used to lookup the description of some keys available in the data sets. 
2.	DIAS Information Levels – Attributes 20017.xlsx<br>
Contains some short information about the columns which are available in the main data set with the people specific data.
3.	Udacity_AZDIAS_052018.csv<br>
Demographics data for the general population of Germany (about 900 000 persons with 366 features (columns)
4.	Udacity_CUSTOMERS_051028.csv<br>
Demographics data for customers of a mail-order company which was already done and contains some information about a customer response (about 192 000 persons with 369 features (columns).
5.	Udacity_MAILOUT_051218_TEST.csv<br>
Demographics data for individuals who were already part of a marketing campaign (about 43 000 persons with 367 features (columns). 
6.	Udacity_MAILOUT_052018_TRAIN.csv<br>
Demographics data for indiviudals who were already part of a marketing campaign (about 43 000 persons with 366 features (columns).<br>
All the files with person data contain the same structure besides some peculiarities. 
The customers file (4) has 3 additional columns which can be omitted because they do not appear in the other files. The mailout files are from the same campaign but split into one part with a response (5) and one part without a response (6). The latter file will be used to evaluate the performance of the model. 

## How the Source code is organized

The coding for all the data handling is extracted into helper python files which are located in subfolder “utils” and are class based or perhaps pseudo class based.
There are classes which are more generic and classes which are project specific: 

**Project specific**<br>

| File | Description |
|------|-------------|
| df_proj.py|data loading and preparing of the dictionaries|
|df_data_prep_02.py| data preprocessing|

**More generic** <br>

| File | Description |
|------|-------------|
|df_data_prep_01.py|data preprocessing|
|df_pca.py|PCA processing|
|df_kmeans.py|KMeans processing|
|sl_segment.py|Supervised Learning processing|
|df_stat.py|Just to print/plot some data frame statisticsg|



The goal is to reuse the more generic files for other projects whereas the project specific files controls the general handling.

##	How to run everything

The following is expected:<br>
- There are 6 files in the “data” folder (s. above)
- The original file “DIAS information Levels – Attributes 2017.xlsx” was manually adapted by adding the column “DataType”. Here the columns are classified as numerical or categorical. That information is used to build up a central column dictionary which hold central information to make the preprocessing. “_CHANGED” was added to the file name to give a hint to the modification. 
- There is a “utils” folder with the following python files: 
df_proj.py, df_data_prep_01.py, df_data_prep_02.py, df_pca.py, df_kmeans.py, sl_segment.py
- All the utils files are executed by the central jupyter notebook “Arvato Project Workbook.ipynb”
- To run the notebook in the Udacity workspace you have to run the bash script “bash pip_install.py”. That command file make some updates of the run time environment (update scikit-learn, install xgboost) 
- Depending on the used runtime either the sklearn.preprocessing.Imputer (local) or sklearn.impute.SimpleImputer (Udacity workspace) has to be used. Currently the workbook contains the SimpleImputer. Just exchange SimpleImputer by Imputer or vice versa if necessary. 
- The BalancedRandomForest classifier needs a certain joblib version which cannot be installed in the Udacity workspace. Please use global parameter GC_ESTIMATORS = ['LOGREG', 'RANFOR', 'XGB'] instead of GC_ESTIMATORS = ['LOGREG', 'RANFOR', 'BALRANFOR', 'XGB']
- There are some other global parameters which are used to control the runtime behavior like the GC_MAC_ROWS which can be used to read only a certain number of rows from the data set (GC_MAX_ROWS=0 -> read all rows wanted).

## Licence, Acknowledgement 
The data soures are protected but it was possible to use them for this project. 
Without that kind willingness it would not be possible to work on the project. Many thanks. 




