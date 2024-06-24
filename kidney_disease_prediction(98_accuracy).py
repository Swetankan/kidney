import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('dataset/kidney_disease.csv')

df.head(20)

df.describe()

df.info()

df.isnull().sum()

sns.heatmap(df.isnull())

df.shape

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['rbc']=lb.fit_transform(df['rbc'])
df['pc']=lb.fit_transform(df['pc'])
df['pcc']=lb.fit_transform(df['pcc'])
df['ba']=lb.fit_transform(df['ba'])
df['htn']=lb.fit_transform(df['htn'])
df['dm']=lb.fit_transform(df['dm'])
df['cad']=lb.fit_transform(df['cad'])
df['appet']=lb.fit_transform(df['appet'])
df['pe']=lb.fit_transform(df['pe'])
df['ane']=lb.fit_transform(df['ane'])
df['classification']=lb.fit_transform(df['classification'])

# Replace Blank values with DataFrame.replace() methods.

print(df)

df['age']=df['age'].fillna(df['age'].mean())

df.replace('\t?', float('nan'), inplace=True)  # Replace '\t?' with NaN

# Convert the relevant columns to float
columns_to_convert = [ 'bp',     'sg',   'al' ,  'su',  'rbc',  'pc',  'pcc' , 'ba'  ,'bgr', 'bu', 'sc', 'sod', 'pot' ,'hemo' ,'pcv' , 'wc' , 'rc' ,'htn',  'dm'  ,'cad',  'appet' , 'pe' , 'ane' , 'classification']  # Replace with the actual column names
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)


from fancyimpute import KNN

knn_imputer = KNN()
df = knn_imputer.fit_transform(df)

df=pd.DataFrame(df,columns=['id',   'age',   'bp',     'sg',   'al' ,  'su',  'rbc',  'pc',  'pcc' , 'ba'  ,'bgr', 'bu', 'sc', 'sod', 'pot' ,'hemo' ,'pcv' , 'wc' , 'rc' ,'htn',  'dm'  ,'cad',  'appet' , 'pe' , 'ane' , 'classification'])

df.head(30)

X=df.drop(df['classification'])
y=df['classification'][0:203]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,mean_absolute_error

print(classification_report(y_test,y_pred))

confusion_matrix(y_test,y_pred)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred1 = rf.predict(X_test)
print(classification_report(y_test,y_pred1))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Confusion matrix heatmap for KNN
cm_knn = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Heatmap - KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion matrix heatmap for Random Forest
cm_rf = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix Heatmap - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Accuracy graph with color
accuracy_knn = knn.score(X_test, y_test)
accuracy_rf = rf.score(X_test, y_test)
accuracy_data = pd.DataFrame({'Model': ['KNN', 'Random Forest'], 'Accuracy': [accuracy_knn, accuracy_rf]})
plt.figure(figsize=(8, 6))
sns.barplot(x='Model', y='Accuracy', data=accuracy_data, palette='tab10')
plt.title('Model Accuracy Comparison')
plt.show()

# Age group distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['age'], bins=20, kde=True, color='green')
plt.title('Age Group Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Blood pressure distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['bp'], bins=20, kde=True, color='red')
plt.title('Blood Pressure Distribution')
plt.xlabel('Blood Pressure')
plt.ylabel('Frequency')
plt.show()
