#!/usr/bin/env python
# coding: utf-8
## By: Sanjeet Pal Singh

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV

print('Library Loaded')


## Just added up the headings in the comma seperated list of cleveland data for the sake of easiness to find the labels.
data = './processed.cleveland.csv'


df = pd.read_csv(data)
df.shape

# For printing the first five values with headers
df.head()

# For Replacing the ? with nan (Missing values)
df['age'].replace('?', np.nan, inplace= True)
df['sex'].replace('?', np.nan, inplace= True)
df['cp'].replace('?', np.nan, inplace= True)
df['trestbps'].replace('?', np.nan, inplace= True)
df['chol'].replace('?', np.nan, inplace= True)
df['fbs'].replace('?', np.nan, inplace= True)
df['restecg'].replace('?', np.nan, inplace= True)
df['thalach'].replace('?', np.nan, inplace= True)
df['exang'].replace('?', np.nan, inplace= True)
df['oldpeak'].replace('?', np.nan, inplace= True)
df['slope'].replace('?', np.nan, inplace= True)
df['ca'].replace('?', np.nan, inplace= True)
df['thal'].replace('?', np.nan, inplace= True)
df['target'].replace('?', np.nan, inplace= True)

df.isna().sum() # To check null value is there or not

# For further replacing the NaN values with 0
df['ca'].replace(np.nan, 0, inplace= True)
df['thal'].replace(np.nan, 0, inplace= True)
df.isna().sum() # To check null value is there or not anymore


cols = df.columns
cols



print("# rows in dataset {0}".format(len(df)))
print("-------------------------------------------")


for col in cols:
    print("# rows in {1} with ZERO value: {0}".format(len(df.loc[df[col] == 0 ]),col))


df.dtypes

# Converting ca and thal values to int64
df.astype({'ca': 'int64', 'thal':'int64'}).dtypes


# Converting restecg positive values where the abnormality is to 1
# Means replacing 2 (showing probable or definite left ventricular hypertrophy) = 1

df['restecg'].replace(2, 1, inplace= True)

# For Replacing positive cases where value is 2,3,4 to 1 in diagnosis or target values
df['target'].replace([2,3,4], 1, inplace= True)




# # Visualization



# For printing Correlation heat matrix plot
# Can be skipped if just want to see the results of the classiciation models just comment the visualization part
corrmat = df.corr()
fig = plt.figure(figsize = (16, 16))
sns.heatmap(corrmat, vmax = 1, square = True,annot=True,vmin=-1)
plt.show()


## For printing histogram
df.hist(figsize=(12,12))
plt.show()


print ('Building bar plot (works only on jupyter notebook) Press Ctrl + C if it stucks for more than 2-3 minutes.\n')
sns.barplot(x="sex", y="age", hue="target", data=df)


# Getting error now for plotting the pairplot need to check the reason don't have time 
# sns.pairplot(df,hue='target',)

print ('Removing the columns sex, target, age to ease the computation\n')
## correcting the mistake
cols=list(cols)
cols.remove('sex')
cols.remove('target')
cols.remove('age')
cols


## Doing the dimension reduction to convert the 14 dimensions into 2 before plotting the scatterplot graph
## Use either TSNE or PCA or both but then need to add the pipeline to pass out of PCA to TSNE or vice versa
print ('Using TSNE\n')
X=df.drop('target',axis=1)
from sklearn.manifold import TSNE
import time
time_start = time.time()

df_tsne = TSNE(random_state=10).fit_transform(X)

print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_tsne

##Uncomment the below multi line comment if you want to see the scattered graph
## can take some time - depends upon amount of data we are feeding and number of cpus on the system
"""import matplotlib.patheffects as PathEffects
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("deep", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts



fashion_scatter(df_tsne, df.target)"""


# # Feature Engineering


print ('Doing Feature Engineering\n')
df.target.value_counts()



print("# rows in dataset {0}".format(len(df)))
print("-------------------------------------------")



for col in cols:
    print("# rows in {1} with ZERO value: {0}".format(len(df.loc[df[col] == 0 ]),col))





X = df.drop('target',axis=1) # predictor feature coloumns
y = df.target


X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.10, random_state = 10)

print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))



from sklearn.preprocessing import Imputer
#impute with mean all 0 readings

fill = Imputer(missing_values = 0 , strategy ="mean", axis=0)

X_train = fill.fit_transform(X_train)
X_test = fill.fit_transform(X_test)


# # Model Building and Evaluation


def FitModel(X_train, y_train, X_test, y_test, algo_name, algorithm, gridSearchParams, cv):
    np.random.seed(10)
   
    print ('Running '+algo_name) 
    grid = GridSearchCV(
        estimator=algorithm,
        param_grid=gridSearchParams,
        cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    
    
    grid_result = grid.fit(X_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    
    print(pred)

    print('Best Params :',best_params)
    print('Classification Report :\n', classification_report(y_test, pred))
    print('Accuracy Score : ' + str(accuracy_score(y_test, pred)))
    print('Confusion Matrix : \n', cm)


#Its better to run the below models one by one by using commenting

# # Logistic Regression

# Create regularization penalty space
penalty = ['l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

#Default value of max_iteration in logistic regression is 100, I did randomly tried 1000
FitModel(X_train, y_train, X_test, y_test, 'LogisticRegression', LogisticRegression(solver='lbfgs', max_iter=1000), hyperparameters, cv=5)
 
 

# # XGBoost - can take lot of time better to comment the code
# !!! Warning XGBoost can take lot of time depends upon number of processors or kind of processor you have ##

param ={
            'n_estimators': [100, 500, 1000,1500, 2000],
            'max_depth' :[2,3,4,5,6,7],
    'learning_rate':np.arange(0.01,0.1,0.01).tolist()
           
        }

FitModel(X_train,y_train,X_test,y_test,'XGBoost',XGBClassifier(),param,cv=5) 
 
 
# # Random Forest

param ={
            'n_estimators': [100, 500, 1000,1500, 2000],
    'max_depth' :[2,3,4,5,6,7],
           
        }
FitModel(X_train, y_train, X_test, y_test, 'Random Forest', RandomForestClassifier(), param, cv=5)


# # SVC

param ={
            'C': [0.1, 1, 100, 1000],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        }

FitModel(X_train, y_train, X_test, y_test, 'SVC', SVC(), param, cv=5)



