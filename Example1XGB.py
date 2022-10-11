#Example 1 is balanced data set, no hyperparameters as splits data into training and testing sets
#Goal is to predict if protein is SNP or PD

#**import XGBoost and other ML modules**
import pandas as pd #import data for One-Hot Encoding
import numpy as np #calc the mean and SD
import xgboost as xgb #XGboost ML
import matplotlib.pyplot as plt #graphing/plotting stuff
from xgboost import XGBClassifier #SK learn API for XGB model building
from xgboost import XGBRegressor #SK learn API for XGB regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split #Splits data frame into the training set and testing set
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer #Scoring metrics 
from sklearn.model_selection import GridSearchCV #Cross validation to improve hyperparameters
from sklearn.metrics import confusion_matrix #creates the confusion matrix - stats on how accurate the test set output is
from sklearn.metrics import ConfusionMatrixDisplay #draws the confusion matrix

#**Create, clean and convert the train.csv dataset to a dataframe**
pd.set_option('display.max_columns', None) #Max columns
df = pd.read_csv('demo.csv') #Pandas creates data frame from the .csv mutation data
df.drop(['pdbcode:chain:resnum:mutation'],axis=1, inplace=True) #removes columns undeeded in training
df.columns = df.columns.str.replace(' ', '_') #Removes gaps in column names
df.replace(' ', '_', regex=True, inplace=True) #Replace all blank spaces with underscore (none were present)

#**Encoding the categorical data for dataframe y**
X = df.drop('dataset', axis=1).copy() #X is dataframe with data used to train and predict if SNP or PD 
y_encoded = pd.get_dummies(df, columns=['dataset'], prefix=['Mutation']) #y is dataframe with mutations in non-object format for testing
y = y_encoded[['Mutation_pd','Mutation_snp']].copy()
print(y)
print(X)

#**Split data into training and test**

#y_count = y.drop(['Binding', 'SProtFT0', 'SProtFT1', 'SProtFT2', 'SProtFT3', 'SProtFT4', 'SProtFT5', 'SProtFT6', 'SProtFT7', 'SProtFT8', 'SProtFT9', 'SProtFT10', 'SProtFT11', 'SProtFT12', 'Interface', 'Relaccess', 'Impact', 'HBonds', 'SPhobic', 'CPhilic', 'BCharge', 'SSGeom', 'Voids', 'MLargest1', 'MLargest2', 'MLargest3', 'MLargest4', 'MLargest5', 'MLargest6', 'MLargest7', 'MLargest8', 'MLargest9', 'MLargest10', 'NLargest1', 'NLargest2', 'NLargest3', 'NLargest4', 'NLargest5', 'NLargest6', 'NLargest7', 'NLargest8', 'NLargest9', 'NLargest10', 'Clash', 'Glycine', 'Proline', 'CisPro'],axis=1)
#print(y_count.count()) #Shows data is already balanced

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y) #Splits data into training and testing (even tho already split)

#**Build XGB training/ classification model**
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42)
clf_xgb.fit(
    X_train,
    y_train,
    early_stopping_rounds =10,
    verbose=True,
    eval_metric='rmse',
    eval_set=[(X_test, y_test)])
#Stop building trees 10 rounds after prediction does not improve. Area Under Curve evalutation. Test set used to decide how many trees to build
print(clf_xgb)










