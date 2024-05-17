#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data=pd.read_csv("artist.csv")
print(data)

print(data.head())

print(data.shape)

print(data['Final'].describe())

plt.plot(data['Final'], color='r')
plt.title('Final Grade', fontsize=20)
plt.xlabel('Final Grade', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.show()

print(data.isnull().any())

#Counting the number of males and females
male_student = len(data[data['sex'] == 'M'])
female_student= len(data[data['sex'] == 'F'])
print('Number of male students:',male_student)
print('Number of female students:',female_student)

#Determines the age with respect to sex
plt.boxplot(data['age'], vert=False, patch_artist=True, boxprops=dict(facecolor='g'))
plt.title('Age Distribution', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Sex', fontsize=15)
plt.show()

plt.plot(data['address'], color='r')
plt.title('Urban vs Rural Students', fontsize=20)
plt.xlabel('Address', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.show()

#dropping the unnecessary columns
data.drop(["school","age"], axis=1, inplace=True)

print(data.head())

data_dum=data

#Converting to categorical value
categorical_d = {'yes': 1, 'no': 0}
data_dum['Astatus'] = data_dum['Astatus'].map(categorical_d)
data_dum['paid'] = data_dum['paid'].map(categorical_d)
data_dum['activities'] = data_dum['activities'].map(categorical_d)


categorical_d = {'F': 1, 'M': 0}
data_dum['sex'] = data_dum['sex'].map(categorical_d)

# map the address data
categorical_d = {'U': 1, 'R': 0}
data_dum['address'] = data_dum['address'].map(categorical_d)

categorical_d = {'I': 1, 'B': 0}
data_dum['status'] = data_dum['status'].map(categorical_d)

categorical_d = {'acrylic': 1, 'watercolor': 0}
data_dum['mode'] = data_dum['mode'].map(categorical_d)

# map the guardian data
categorical_d = {'mother': 0, 'father': 1, 'other': 2}
data_dum['guardian'] = data_dum['guardian'].map(categorical_d)

print(data_dum)

print(data_dum.columns)

from sklearn.model_selection import train_test_split
x=data_dum.drop("Final",axis=1)
y=data_dum['Final']

print(data_dum['Final'])

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=44)

from sklearn.linear_model import LinearRegression 

L=LinearRegression()

L.fit(X_train, y_train)
pickle.dump(L, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
y_pred=L.predict(X_test)

print(L.score(X_test, y_test)) 

plt.plot(data['P(out of 5)'], data['Final'], color='g')
plt.xlabel('Performance', fontsize=15)
plt.ylabel('Final Score', fontsize=15)
plt.show()

