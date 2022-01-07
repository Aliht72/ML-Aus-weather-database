import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
                                                                                #Import libraries which we use
start = time.time()
Weather = pd.read_csv('/Users/ali/Desktop/Rain in Aus/weatherAUS.csv')
Weather['Date'] = pd.to_datetime(Weather['Date']) 
#Change data of date format to date 
Weather['Year'] = Weather['Date'].dt.year
Weather['Month'] = Weather['Date'].dt.month
Weather['Day'] = Weather['Date'].dt.day
Weather.drop('Date', axis=1, inplace = True)
#Delete Data column and change it to Year, Month and Day with int format
#print(Weather[categorical].isnull().sum())
pd.get_dummies(Weather.WindDir9am, drop_first=True, dummy_na=True)
pd.get_dummies(Weather.WindGustDir, drop_first=True, dummy_na=True)
pd.get_dummies(Weather.WindDir3pm, drop_first=True, dummy_na=True)
pd.get_dummies(Weather.RainToday, drop_first=True, dummy_na=True)
                                                                            #Encode catagorical variables to binary variables

#print(Weather[numerical].isnull().sum())
sns.boxplot(y="Rainfall", data=Weather)
plt.show()

sns.boxplot(y="Evaporation", data=Weather)
plt.show()

sns.boxplot(y="WindSpeed9am", data=Weather)
plt.show()

sns.boxplot(y="WindSpeed3pm", data=Weather)
plt.show()

plt.hist(Weather['WindSpeed9am'], bins=10, rwidth=0.99)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('WindSpeed9am')
plt.show()

plt.hist(Weather['WindSpeed3pm'], bins=10, rwidth=0.99)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('WindSpeed3pm')
plt.show()

plt.hist(Weather['Evaporation'], bins=10, rwidth=0.99)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Evaporation')
plt.show()

plt.hist(Weather['Rainfall'], bins=10, rwidth=0.99)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Rainfall')
plt.show()
correlation = Weather.corr()

plt.figure(figsize=(14,10))
plt.title('Correlation Heatmap of Rain in Australia Dataset')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.show()
num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']
sns.pairplot(Weather[num_var], kind='scatter', diag_kind='hist', palette='Rainbow')
plt.show()
                                                                    #Show the plot and histogram of numeric variables
for Weather in [Weather]:
    Weather['RainTomorrow'].fillna(Weather['RainTomorrow'].mode()[0], inplace=True)
X = Weather.drop(['RainTomorrow'], axis=1)
y = Weather['RainTomorrow']
print(Weather['RainTomorrow'].unique())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
                                                                        #Seperate the dataset into to the two part, Train and Test
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
X_train[numerical].isnull().sum()
for col in numerical:
    if X_train[col].isnull().mean()>0:
        (col, round(X_train[col].isnull().mean(),4))

for weather1 in [X_train, X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        weather1[col].fillna(col_median, inplace=True)
print(X_test[numerical].isnull().sum())
                                                                        #Impute NA variables of numeric data with their median
for Weather2 in [X_train, X_test]:
    Weather2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    Weather2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    Weather2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    Weather2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
                                                                            #Impute NA variables of categorical data with the most frequent variables of each one
def max_value(weather3, variable, top):
    return np.where(weather3[variable]>top, top, weather3[variable])
for Weather3 in [X_train, X_test]:
    Weather3['Rainfall'] = max_value(Weather3, 'Rainfall', 3.2)
    Weather3['Evaporation'] = max_value(Weather3, 'Evaporation', 21.8)
    Weather3['WindSpeed9am'] = max_value(Weather3, 'WindSpeed9am', 55)
    Weather3['WindSpeed3pm'] = max_value(Weather3, 'WindSpeed3pm', 57)
                                                                    #Impute the outliars of this four numerica variables with the calculating IQR

encoder = ce.BinaryEncoder(cols=['RainToday'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
                                                                    #Encoding and spliting RainToday variables to the two variable "Yes" or "No"

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
    pd.get_dummies(X_train.Location), 
    pd.get_dummies(X_train.WindGustDir),
    pd.get_dummies(X_train.WindDir9am),
    pd.get_dummies(X_train.WindDir3pm)], axis=1)

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
    pd.get_dummies(X_test.Location), 
    pd.get_dummies(X_test.WindGustDir),
    pd.get_dummies(X_test.WindDir9am),
    pd.get_dummies(X_test.WindDir3pm)], axis=1)
                                                                    #Unifying all variables

cols = X_train.columns

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])
                                                                #Scaling the numeric variables

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear', random_state=0)
                                                                #Implementing Logistic Regression on our dataset

logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test)

y_pred_test

logreg.predict_proba(X_test)[:,0]

logreg.predict_proba(X_test)[:,1]

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

y_pred_train = logreg.predict(X_train)

y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print('Training set score: {:.4f}'.format(logreg.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)

logreg100.fit(X_train, y_train)

print('Training set score: {:.4f}'.format(logreg100.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01, solver='liblinear', random_state=0)

logreg001.fit(X_train, y_train)

print('Training set score: {:.4f}'.format(logreg001.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(logreg001.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix , classification_report
print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
                                                                                #Calculating accuracy and creating matrix of confusion


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
    index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()
                                                                                #Heatmap of confusion matrix

end = time.time()
print(f"Runtime of the program is {end - start}")
                                                                                #calculating the time of process