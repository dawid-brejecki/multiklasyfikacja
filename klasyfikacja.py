

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
import xgboost
import lightgbm as lgb 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import SGDClassifier



df=pd.read_csv('bodyPerformance.csv')

# preprocessing
df = df.dropna()

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['class'] = le.fit_transform(df['class'])


column_names_to_not_normalize = ['gender', 'class']
column_names_to_normalize = [x for x in list(df) if x not in column_names_to_not_normalize]

scaler = StandardScaler()
df[column_names_to_normalize] = scaler.fit_transform(df[column_names_to_normalize])


target = df.pop('class')
X_train, X_test, Y_train, Y_test = train_test_split(df, target, test_size = 0.25)

sns.set_theme(style="ticks")
sns.pairplot(df)



df.head()
df.info()

# wykres przykladowej zmiennej
sns.set()
plt.figure(figsize=(10, 5))
plt.title('Age Distribution')
sns.distplot(df['body fat_%'])
plt.show()
df.isnull().sum()




# 1. Drzewo decyzyjne

# metoda przeszukiwania siatki, takze z walidacja krzyzowa

classifier = DecisionTreeClassifier()
params = {'criterion': ['gini', 'entropy'],
          'max_depth': np.arange(1, 30),
         'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}

grid_search = GridSearchCV(classifier, param_grid=params, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)
grid_search.best_estimator_
grid_search.score(X_test, Y_test)

lista1 = []
lista2 = []




start_time = time.time()
classifier = DecisionTreeClassifier(**grid_search.best_params_)
classifier.fit(X_train, Y_train)

# predykcja
Y_pred = classifier.predict(X_test)
end_time = time.time()

# wyniki
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')

lista1.append(accuracy_score(Y_test, Y_pred))
lista2.append(end_time-start_time)


# 2. Las losowy

classifier = RandomForestClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': np.arange(4, 20),
    'min_samples_leaf': [6, 10, 15],
    'n_estimators': [50]
}

grid_search = GridSearchCV(classifier, param_grid=param_grid, n_jobs=-1, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)
grid_search.best_params_

start_time = time.time()
classifier = RandomForestClassifier(**grid_search.best_params_)
classifier.fit(X_train, Y_train)

# predykcja
Y_pred = classifier.predict(X_test)
end_time = time.time()

# wyniki
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')

lista1.append(accuracy_score(Y_test, Y_pred))
lista2.append(end_time-start_time)




# 3. K-najblizszych sąsiadów

classifier = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(3,20)
}
grid_search = GridSearchCV(classifier, param_grid=param_grid, n_jobs=-1, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)
grid_search.best_params_

start_time = time.time()
classifier = KNeighborsClassifier(**grid_search.best_params_)
classifier.fit(X_train, Y_train)

# predykcja
Y_pred = classifier.predict(X_test)
end_time = time.time()

# wyniki
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')

lista1.append(accuracy_score(Y_test, Y_pred))
lista2.append(end_time-start_time)



# 4. Maszyna wektorow nosnych

classifier = SVC(random_state = 42)

param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': np.arange(0.5,4)
}

grid_search = GridSearchCV(classifier, param_grid=param_grid, n_jobs=-1, scoring='accuracy', cv=5)
grid_search.fit(X_train, Y_train)
grid_search.best_params_

start_time = time.time()
classifier = SVC(**grid_search.best_params_)
classifier.fit(X_train, Y_train)

# predykcja
Y_pred = classifier.predict(X_test)
end_time = time.time()

# wyniki
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')

lista1.append(accuracy_score(Y_test, Y_pred))
lista2.append(end_time-start_time)





# 5. Regresja logistyczna

start_time = time.time()
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_test)
end_time = time.time()

cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')

lista1.append(accuracy_score(Y_test, Y_pred))
lista2.append(end_time-start_time)





# 6. XGBoost

classifier = XGBClassifier()

param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1.5, 5],
        'subsample': [0.6, 1.0],
        'colsample_bytree': [0.6, 1.0],
        'max_depth': [4, 7]}

grid_search = GridSearchCV(classifier, param_grid=param_grid, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, Y_train)
grid_search.best_params_

start_time = time.time()
classifier = XGBClassifier(**grid_search.best_params_)
classifier.fit(X_train, Y_train)

# predykcja
Y_pred = classifier.predict(X_test)
end_time = time.time()

# wyniki
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')

lista1.append(accuracy_score(Y_test, Y_pred))
lista2.append(end_time-start_time)





# 7. Light GBM

classifier = lgb.LGBMClassifier()

param_grid = {
    'learning_rate': [0.005, 0.01],
    'objective': ['multiclass'],
    'metric': ['multi_logloss'],
     'num_class': [4]
}

grid_search = GridSearchCV(classifier, param_grid=param_grid, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, Y_train)
grid_search.best_params_

param = {**grid_search.best_params_}

start_time = time.time()
train_data=lgb.Dataset(X_train,label=Y_train)
num_round=50
lgbm=lgb.train(param, train_data,num_round)

Y_pred=lgbm.predict(X_test)
Y_pred = [np.argmax(line) for line in Y_pred]
end_time = time.time()


# wyniki
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')

lista1.append(accuracy_score(Y_test, Y_pred))
lista2.append(end_time-start_time)





# 8. SGD Classifier

clf = SGDClassifier()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# wyniki
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')





# poprawiamy recznie skutecznosc klasyfikatora

losses = ["hinge", "log", "modified_huber", "perceptron", "squared_hinge"]
scores = []
for loss in losses:
    clf = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)
    clf.fit(X_train, Y_train)
    scores.append(clf.score(X_test, Y_test))
  
plt.title("Effect of loss")
plt.xlabel("loss")
plt.ylabel("score")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores) 





n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    clf = SGDClassifier(loss="log", penalty="l2", max_iter=n_iter)
    clf.fit(X_train, Y_train)
    scores.append(clf.score(X_test, Y_test))
  
plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores) 





clf = SGDClassifier(loss="log", max_iter=50)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# wyniki
cm = confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}', f'Time: {end_time-start_time}')

lista1.append(accuracy_score(Y_test, Y_pred))
lista2.append(end_time-start_time)
# nie udalo sie poprawic skutecznosci klasyfikatora





nazwy = ['drzewo decyzyjne', 'las losowy', 'K-najblizszych sasiadow', 'maszyna wektorow nosnych',
        'regresja logistyczna', 'XGBoost', 'LightGBM', 'SGD']
d = {'klasyfikator': nazwy,
     'wsp. determinancji': lista1,
     'czas wyk. kodu': lista2}
df = pd.DataFrame(d)
df = df.sort_values('wsp. determinancji', ascending = False).reset_index(drop=True)
print(df)





# najdokładniejszym klasyfikatorem okazał się XGBoost, jednakże czas jego implementacji także był najdłuższy.
# skuteczność modeli w dużej mierze zależy od ich "tuningowania" oraz ochrony przed nadmiernym przeuczeniem.
# na podstawie wyników można jednak przyjąć, że z powodzeniem w klasyfikacji tych danych poradzą sobie klasyfikatory XGBoost,
# las losowy, SVM oraz lightGBM.
# najtrudniejsze do rozróżnienia okazały się dwie środkowe klasy.






