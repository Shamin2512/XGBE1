#Example 1 is balanced data set, no hyperparameters as splits data into training and testing sets
#Goal is to predict if protein is SNP or PD

#**import XGBoost and other ML modules**
import pandas as pd #import data for One-Hot Encoding
import numpy as np #calc the mean and SD
import xgboost as xgb #XGboost Learning API
import matplotlib.pyplot as plt #graphing/plotting stuff
from xgboost import XGBClassifier #SK learn API for XGB model building
from xgboost import XGBRegressor #SK learn API for XGB regression
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #Splits data frame into the training set and testing set
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer #Scoring metrics 
from sklearn.model_selection import GridSearchCV #Cross validation to improve hyperparameters
from sklearn.metrics import confusion_matrix #creates the confusion matrix - stats on how accurate the test set output is
from sklearn.metrics import ConfusionMatrixDisplay #draws the confusion matrix

xgb.set_config(verbosity=2)
config = xgb.get_config()
print(config)
          
#**Create, clean and convert the train.csv dataset to a dataframe**
df = pd.read_csv('demo.csv') #Pandas creates data frame from the .csv mutation data
df.drop(['pdbcode:chain:resnum:mutation'],axis=1, inplace=True) #removes columns unrequired columns, replacing the variable 
df.columns = df.columns.str.replace(' ', '_') #Removes gaps in column names
df.replace(' ', '_', regex=True, inplace=True) #Replace all blank spaces with underscore (none were present)

#**Encoding the categorical data for dataframe y**
X = df.drop('dataset', axis=1).copy() #X is dataframe with data used to train and predict if SNP or PD 
y_encoded = pd.get_dummies(df, columns=['dataset'], prefix=['Mutation']) #y is df with mutations changing from object -> unint8 (integer)
y = y_encoded['Mutation_pd'].copy().astype('int32') #datafram y only has one column of mutation data; binary

#**Split data into training and test**
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=42, stratify=y) #Splits data into training and testing

#**XGB Dmatrix training model**
d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)

param = { #Dictionary of parameters to initally train the model
    'booster': 'gbtree', #non-linear, tree method (default)
    'verbosity': 1, #outputs the evaluation of each tree
    'eta': 0.1, #Same as learning rate, shrinkage of each step when approaching the optimum value
    'colsample_bytree': 0.8, #How much subsampling for each tree
    'max_depth': 5, #Greater the depth, more prone to overfitting; tune from CV
    'eval_metric': ['auc', 'aucpr'],
    'min_child_weight': 1,
    'objective': 'binary:logistic' #classifies the outcome as either 0 (SNP), or 1 (PD). Non multiclass classification
    }

model = xgb.train(param, d_train, evals= [(d_test, 'eval'), (d_train, 'train')], num_boost_round=50, early_stopping_rounds=20)

#Cross validation paramaters
dmatrix_val = xgb.DMatrix(X, y)
params = {
    'objective': 'binary:hinge',
    'colsample_bytree': 0.3,
    'eta': 0.1,
    'max_depth': 3
}
cross_val = xgb.cv(
    params=params,
    dtrain=dmatrix_val, 
    nfold=5,
    num_boost_round=50, 
    early_stopping_rounds=10, 
    metrics='error', 
    as_pandas=True,
    seed=42
    )
print(cross_val.head())

#**Plot confusion matrix using the true and predicted values**
#clf = xgb.XGBClassifier(**param)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

y_pred = model.predict(d_test) #No longer a pandas DF, is now a numpy array as Dmatrix
cm = confusion_matrix(y_test, y_pred > 0.5)

print("Confusion Matrix:\n", cm)
print("MCC:\n", matthews_corrcoef(y_test.astype(int), y_pred>0.5))
print("F1 Score:\n", f1_score(y_test.astype(int), y_pred>0.5))



 








