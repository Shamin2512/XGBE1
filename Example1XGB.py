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

#**One-Hot Encoding the categorical data for dataframe y**
X = df.drop('dataset', axis=1).copy() #Dataframe with data used to train and predict if SNP or PD
#X = X_drop.head(50000) #Takes first 50000 rows as datapoints for training since data is random order
#X.to_csv('xtrain.csv') #Output .csv

#y_tail = df.tail(11205)
y_encoded = pd.get_dummies(df, columns=['dataset'], prefix=['Mutation']) #Dataframe with mutations in numerical categorical format for testing
y = y_encoded[['Mutation_pd','Mutation_snp']].copy() #y is dataframe with only mutation columns for testing against
#y.to_csv('ytest.csv')

#**Split data into training and test**

#y_count = y.drop(['Binding', 'SProtFT0', 'SProtFT1', 'SProtFT2', 'SProtFT3', 'SProtFT4', 'SProtFT5', 'SProtFT6', 'SProtFT7', 'SProtFT8', 'SProtFT9', 'SProtFT10', 'SProtFT11', 'SProtFT12', 'Interface', 'Relaccess', 'Impact', 'HBonds', 'SPhobic', 'CPhilic', 'BCharge', 'SSGeom', 'Voids', 'MLargest1', 'MLargest2', 'MLargest3', 'MLargest4', 'MLargest5', 'MLargest6', 'MLargest7', 'MLargest8', 'MLargest9', 'MLargest10', 'NLargest1', 'NLargest2', 'NLargest3', 'NLargest4', 'NLargest5', 'NLargest6', 'NLargest7', 'NLargest8', 'NLargest9', 'NLargest10', 'Clash', 'Glycine', 'Proline', 'CisPro'],axis=1)
#print(y_count.count()) #Shows data is already balanced

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.81694, random_state=42, stratify=y) #Splits data into training and testing (even tho already split)

#**Build XGB training/ classification model**
clf = xgb.XGBClassifier(objective='binary:logistic', seed=42)
clf.fit(
    X_train,
    y_train,
    early_stopping_rounds=10,
    verbose=True,
    eval_metric='rmse',
    eval_set=[(X_test, y_test)]) #Rmse metric, early stopping rounds incompatible with fit()
print(clf)


#**Plot confusion matrix using the true and predicted values**
y_pred = clf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test.values.argmax(axis=1), y_pred.argmax(axis=1), values_format='d', display_labels=["PD", "SNP"])
print(confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))







