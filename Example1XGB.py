#Example 1 is balanced data set, no hyperparameters as splits data into training and testing sets
#Goal is to predict if protein is SNP or PD

#**import XGBoost and other ML modules**
import pandas as pd #import data for One-Hot Encoding
import numpy as np #calc the mean and SD
from xgboost import cv #Cross validation 
import xgboost as xgb #XGboost ML
import matplotlib.pyplot as plt #graphing/plotting stuff
from xgboost import XGBClassifier #SK learn API for XGB model building
from xgboost import XGBRegressor #SK learn API for XGB regression
from sklearn.metrics import accuracy_score
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
y_encoded = pd.get_dummies(df, columns=['dataset'], prefix=['Mutation']) #Dataframe with mutations in numerical categorical format for testing
y = y_encoded[['Mutation_pd','Mutation_snp']].copy() #y is dataframe with only mutation columns for testing against


#**Converting the pandas dataframe to a XGB Dmatrix**
data_dmatrix = xgb.DMatrix(data= X, label =y)


#**Split data into training and test**
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.81694, random_state=42, stratify=y) #Splits data into training and testing, specifying the training set should be 50,000 rows


#**Build XGB training/ classification model**
params = {
    'objective':"binary:logistic",
    'max_depth': 4,
    'alpha': 10,
    'learning_rate': 1.0,
    'n_estimators': 100,
    'seed': 42,
    }

clf = xgb.XGBClassifier(**params)
clf.fit(
    X_train,
    y_train,
    )
print(clf)

y_pred = clf.predict(X_test)
print('XGBoost model accuracy score: ', format(accuracy_score(y_test, y_pred)))


nparams = {
    'objective':"binary:logistic",
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha' :10
    }

xgb_cv =cv(
    dtrain=data_dmatrix,
    params=nparams,
    nfold=3,
    num_boost_round=50,
    early_stopping_rounds=10,
    metrics='rmse',
    as_pandas=True,
    seed=42
    )

xgb.plot_importance(clf) #Feature importance

#**Plot confusion matrix using the true and predicted values**
y_pred = clf.predict(X_test).argmax(axis=1)
y_test = y_test.values.argmax(axis=1) #converts the encoded arrray to integer list
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, values_format='d', display_labels=["PD", "SNP"])
print(confusion_matrix(y_test, y_pred))







