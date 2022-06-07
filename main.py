from fileinput import filename
from telnetlib import BM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#---loading the dataset and printing the information about dataframe---
df = pd.read_csv('diabetes.csv')
df.info()
print("----------")

#---checking the null values---
print("Null values")
print(df.isnull().sum())
print("----------")

#---checking for 0s---
print("0 values")
print(df.eq(0).sum())
print("----------")

#---replacing 0 values with NaN---
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]=\
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)

#---replacing NaN values with mean of each column---
df.fillna(df.mean(), inplace=True)
print(df.eq(0).sum())

#---examining various independent features affect the outcome---
corr=df.corr()
print(corr)
 
 #---plottting the results returned by the corr() function as a matrix
fig,ax=plt.subplots(figsize=(10,10))
cax=ax.matshow(corr,cmap='coolwarm',vmin=-1,vmax=1)

fig.colorbar(cax)
ticks=np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)

ax.set_xticklabels(df.columns)
plt.xticks(rotation=90)

ax.set_yticklabels(df.columns)
ax.set_yticks(ticks)

#---printing the correlation factor---
for i in range(df.shape[1]):
    for j in range(9):
        text=ax.text(j,i,round(corr.iloc[i][j],2),ha="center",va="center",color="w")
plt.show

#---using heatmap to show the correlation matrix---
#import seaborn as sns
#sns.heatmap(df.corr(),annot=True)
#fig=plt.gcf()
#fig.set_size_inches(8,8)

#---getting the top 4 features that has the heighest correlation---
print(df.corr().nlargest(4,'Outcome').index)

#---printing top 4 correlation values---
print(df.corr().nlargest(4,'Outcome').values[:,8])

print("----------")

#   LOGISTIC REGRESSION
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

#---features---
X=df[['Glucose','BMI','Age']]
#print(X.shape)

#---label---
Y=df.iloc[:,8]
#print(Y.shape)

log_regress=linear_model.LogisticRegression()
log_regress_score=cross_val_score(log_regress,X,Y,cv=10,scoring='accuracy').mean()

#print("Logistic Regression Score -- ",log_regress_score)

result = []
result.append(log_regress_score)

#   K-NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier

#---empty list that will hold cross validation scores---
cv_scores=[]
folds=10
ks=list(range(1,int(len(X)*((folds-1)/folds)),2))
for k in ks:
    knn=KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(knn,X,Y,cv=folds,scoring='accuracy').mean()
    cv_scores.append(score)

knn_score=max(cv_scores)
optimal_k=ks[cv_scores.index(knn_score)]

#print(f"the optimal nmberof neighbors is {optimal_k}")
#print("KNN Score -- ",knn_score)

result.append(knn_score)

#   SUPPORT VECTOR MACHINES
from sklearn import svm

linear_svm=svm.SVC(kernel='linear')
linear_svm_score=cross_val_score(linear_svm,X,Y,cv=10,scoring='accuracy').mean()

#print("SVM Score -- ",linear_svm_score)
result.append(linear_svm_score)

#--- using RBF kernal---
rbf=svm.SVC(kernel='rbf')
rbf_score=cross_val_score(rbf,X,Y,cv=10,scoring='accuracy').mean()
#print("RBF Scorre -- ",rbf_score)
result.append(rbf_score)

#   Selecting the Best Model
algorithms=["LOGISTIC REGRESSION","K-NEAREST NEIGHBORS","SUPPORT VECTOR MACHINES","SVM RBF KERNEL"]
cv_mean=pd.DataFrame(result,index=algorithms)
cv_mean.columns=["Accuracy"]
cv_mean.sort_values(by="Accuracy",ascending=False)
print(cv_mean)


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV



#Create copy of dataset.
df_model = df.copy()
#Rescaling features age, trestbps, chol, thalach, oldpeak.
scaler = StandardScaler()
features = [['Glucose', 'BMI', 'Age']]
for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])
#Create KNN Object
knn = KNeighborsClassifier()
#Create x and y variable
x = df_model.drop(columns=['Outcome'])
y = df_model['Outcome']
#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
#Training the model
knn.fit(x_train, y_train)
#Predict testing set
y_pred = knn.predict(x_test)

print("-----Standard Scaling-----")
#Check performance using accuracy
print("Accuracy -- ",accuracy_score(y_test, y_pred))
#Check performance using roc
print("ROC -- ",roc_auc_score(y_test, y_pred))



#Create copy of dataset.
df_model1 = df.copy()
#Rescaling features age, trestbps, chol, thalach, oldpeak.
scaler = RobustScaler()
features = [['Glucose', 'BMI', 'Age']]
for feature in features:
    df_model1[feature] = scaler.fit_transform(df_model[feature])
#Create KNN Object
knn = KNeighborsClassifier()
#Create x and y variable
x = df_model1.drop(columns=['Outcome'])
y = df_model1['Outcome']
#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
#Training the model
knn.fit(x_train, y_train)
#Predict testing set
y_pred = knn.predict(x_test)

print("-----Robust Scaling-----")
#Check performance using accuracy
print("Accuracy -- ",accuracy_score(y_test, y_pred))
#Check performance using roc
print("ROC Score -- ",roc_auc_score(y_test, y_pred))






#---Training and Saving the Model
#knn=KNeighborsClassifier(n_neighbors=19)
#knn.fit(X,Y)

#import pickle

#---Saving model to disk---
#filename='diabetes.sav'

#---write to file using write and binary mode---
#pickle.dump(knn,open(filename,'wb'))

#---loading themodel ffrom the disk---
#loaded_model=pickle.load(open(filename,'rb'))

#Glucose=65
#BMI=70
#Age=50

#prediction=loaded_model.predict([[Glucose,BMI,Age]])
#print(prediction)
#if(prediction[0]==0):
#    print("Non-Diabetic")
#else:
#    print("Diabetic")

#prob=loaded_model.predict_proba([[Glucose,BMI,Age]])
#print(prob)
#print("Confidence: "+str(round(np.amax(prob[0])*100,2))+"%")
