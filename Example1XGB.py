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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #Splits data frame into the training set and testing set
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer #Scoring metrics 
from sklearn.model_selection import GridSearchCV #Cross validation to improve hyperparameters
from sklearn.metrics import confusion_matrix #creates the confusion matrix - stats on how accurate the test set output is
from sklearn.metrics import ConfusionMatrixDisplay #draws the confusion matrix

#**Create, clean and convert the train.csv dataset to a dataframe**
df = pd.read_csv('demo.csv') #Pandas creates data frame from the .csv mutation data
df.drop(['pdbcode:chain:resnum:mutation'],axis=1, inplace=True) #removes columns unrequired columns, replacing the variable 
df.columns = df.columns.str.replace(' ', '_') #Removes gaps in column names
df.replace(' ', '_', regex=True, inplace=True) #Replace all blank spaces with underscore (none were present)

#**Encoding the categorical data for dataframe y**
X = df.drop('dataset', axis=1).copy() #X is dataframe with data used to train and predict if SNP or PD 
y_encoded = pd.get_dummies(df, columns=['dataset']) #y is df with mutations changing from object -> unint8 (integer)
y = y_encoded['dataset_pd'].copy()

#**Split data into training and test**
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.81694, random_state=42, stratify=y) #Splits data into training (81.694%) and testing (18.216%). 

#**Build XGB training/ classification model**
clf = xgb.XGBClassifier(objective='binary:logistic', seed=42)
clf.fit(
    X_train,
    y_train,
    early_stopping_rounds=10,
    verbose=True,
    eval_metric='aucpr',
    eval_set=[(X_test, y_test)]) #AUCPR metric, early stopping rounds incompatible with fit()
print(clf)


#**Plot confusion matrix using the true and predicted values**
y_pred = clf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))






