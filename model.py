import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR 
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import confusion_matrix as cm
import pickle

data=pd.read_csv("train.csv",index_col=0)
test_data=pd.read_csv("test.csv",index_col=0)


data.isnull().sum()

q1=data['Survived'].quantile(0.25)
q3=data['Survived'].quantile(0.75)
iqr=q3-q1
upper_limit=q3+1.5*iqr
lower_limit=q1-1.5*iqr
upper_limit,lower_limit


imputer=SimpleImputer(strategy='mean')
mode_imputer=SimpleImputer(strategy='most_frequent')
del_data=data.dropna(axis=0,subset=['Age','Cabin'])
del_data
test_data['Age']=imputer.fit_transform(test_data['Age'].values.reshape(-1,1))
test_data['Fare']=mode_imputer.fit_transform(test_data['Fare'].values.reshape(-1,1))


test_data.isnull().sum()


red_data=del_data.drop(columns=['Name'],axis=1)
red_data
red_test_data=test_data.drop(columns=['Name'],axis=1)


#data['Age'].max()  # 0.42 to 80

red_data['female']=np.where(red_data['Sex']=='female','Yes','No')
red_test_data['female']=np.where(test_data['Sex']=='female','Yes','No')


red_data['female_1']=np.where(red_data['female']=='Yes',1,0)
red_test_data['female_1']=np.where(red_test_data['female']=='Yes',1,0)



numerical_cols=['Pclass','Age','SibSp','Parch','Fare','female_1']
y=red_data['Survived']
x=red_data[numerical_cols]
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3)
test_data_3=red_test_data[numerical_cols]


test_data_3.isnull().sum()


imputer=SimpleImputer()
#x_train=imputer.fit_transform(x_train) 
#x_test=imputer.transform(x_test)
drop_x_train=x_train.select_dtypes(exclude=['object'])
drop_x_test=x_test.select_dtypes(exclude=['object'])


model_1=RandomForestClassifier(random_state=101,n_estimators=100)
model_1.fit(drop_x_train,y_train)
pred_1=model_1.predict(drop_x_test)
print(mae(pred_1,y_test))

scaler=StandardScaler()
drop_x_train=scaler.fit_transform(x_train)
drop_x_test=scaler.transform(x_test)

model_2=LogisticRegression(random_state=0)
model_2.fit(drop_x_train,y_train)
preds_2=model_2.predict(drop_x_test)
print(mae(preds_2,y_test))


model_5=BC(estimator=LR(),n_estimators=150,random_state=42)
model_5.fit(drop_x_train,y_train)
preds_5=model_5.predict(drop_x_test)
#print(preds_5)
print(mae(preds_5,y_test))
print(cm(preds_5,y_test))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix as cm
model_3=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
model_3.fit(drop_x_train,y_train)
preds_3=model_3.predict(drop_x_test)
#print(preds)
print(mae(preds_3,y_test))
print(cm(preds_3,y_test))


import xgboost
from xgboost import XGBClassifier
model_4=XGBClassifier(n_estimators=500,learning_rate=0.05)
model_4.fit(drop_x_train,y_train,early_stopping_rounds=5, 
             eval_set=[(drop_x_test, y_test)],
             verbose=False)
preds_4=model_4.predict(drop_x_test)
print(mae(y_test,preds_4))
print(cm(y_test,preds_4))


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import confusion_matrix as cm
model_6=AdaBoostClassifier(random_state=0,n_estimators=4)
model_6.fit(drop_x_train,y_train)
preds_6=model_6.predict(drop_x_test)
print(mae(preds_6,y_test))
print(cm(preds_6,y_test))



from sklearn.svm import SVC
model_7=SVC(kernel='linear',random_state=101)
model_7.fit(drop_x_train,y_train)
preds_7=model_7.predict(drop_x_test)
print(mae(preds_7,y_test))
print(cm(preds_7,y_test))


from sklearn.ensemble import VotingClassifier  # stacking can also be referred to as a voting
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score 
model=VotingClassifier(estimators=[('RFC',model_1),('LR',model_2),('KNeighborsClassifier',model_3),('XGBClassifier',model_4),('BC',model_5),('AdaBoostClassifier',model_6),('SVC',model_7)],voting='hard')
model.fit(drop_x_train,y_train)
preds=model.predict(drop_x_test)
print(mae(preds,y_test))
print(cm(preds,y_test))
print(model.score(drop_x_test,y_test))
print(accuracy_score(preds,y_test))


submission_preds=model.predict(test_data_3)
pickle.dump(model,open('final.pkl','wb'))

# ab=pd.read_csv("gender_submission.csv")
# ab.iloc[:,0]

# df=pd.DataFrame({"PassegerId":ab.iloc[:,0],"Survived":submission_preds})
# df.reset_index(drop=True)

# df.to_csv("submission.csv",index=False)