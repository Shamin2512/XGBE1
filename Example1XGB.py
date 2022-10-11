#Example 1 is balanced data set, no hyperparameters as splits data into training and testing sets

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
df = pd.read_csv('train.csv') #Pandas creates data frame from the .csv mutation data
print(df)
#df.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason', 'Count', 'Country', 'State', 'Lat Long', 'CustomerID'],
#axis=1, inplace=True) #"Drop" removes the columns listed above as not needed in training. Inplace "True" modifies the data fram itself
