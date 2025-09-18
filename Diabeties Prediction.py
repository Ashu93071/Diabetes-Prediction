#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle,joblib

df=pd.read_csv("C:\\Users\\rushi\\OneDrive\\Desktop\\DATA SCIENCE\\Data sets\\regression data set\\Logistic\\diabetes\\diabetes-dataset.csv")
print(df.isnull().sum()) #checking for null values

#Visualizing the distribution of each feature
# for col in df:
#     plt.hist(df[col])
#     plt.title(col)
#     plt.show()
print(df.describe())
print(df.shape)

##Checking for outliers for each column
# for col in df:
#     plt.boxplot(df[col])
#     plt.title(col)  
#     plt.show()

#1.removing outliers using IQR method
def remove_outliers_iqr(df):
    data=df.copy()
    for col in df:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data= df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return data

clean_data=remove_outliers_iqr(df)
# for col in clean_data:
#     plt.boxplot(clean_data[col])
#     plt.title(col)  
#     plt.show()



#Splitting the data into independent and dependent features
X=clean_data.drop(columns=['Outcome'])
y=clean_data['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

Algo=[LogisticRegression(),RandomForestClassifier(),KNeighborsClassifier()]

#Getting best model with above our set accuracy
# def best_model(Algo,Min_acc):
#     for m in Algo:
#         model=m
#         model.fit(X_train,y_train)
#         re=m.predict(X_test)
#         report=classification_report(y_test,re)
#         acc=accuracy_score(y_test,re)
#         if acc>=Min_acc:
#             Best_model=model
#             return Best_model,acc
# model=best_model(Algo,0.90)

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
result=rf.predict(X_test)
print(accuracy_score(result,y_test))

#Checkin whether our model perfrom good or not.
# print(rf.predict([[2, 130, 2,  1,  80,  44.2 ,0.340, 31]])) 

#SAVE
filename='SaveModel.sav'
#1.save&load our model using pickle
# pickle.dump(rf,open(filename),'wb')
# #load 
# pickle.load(open(filename),'rb')

#2.save and load using joblib
joblib.dump(rf,filename)
joblib.load(filename)
print(df.head(10))