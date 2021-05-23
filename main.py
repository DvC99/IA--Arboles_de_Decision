# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import tree #For our Decision Tree
from sklearn import metrics #For our Decision Tree
from sklearn.model_selection import train_test_split#split in data
import numpy as np
import graphviz 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# csv_file_list = ["winequality-red.csv", "winequality-white.csv"]
# list_of_dataframes = []
# for filename in csv_file_list:
#     list_of_dataframes.append(pd.read_csv(filename))
# merged_df = pd.concat(list_of_dataframes)
# merged_df.to_csv("combined_wine_csv.csv")
np.random.seed(1456)# semilla aleatoria
#data = pd.read_csv('combined_wine_csv.csv', sep=';')
url = 'https://raw.githubusercontent.com/johansbg/DataSetIA/main/combined_wine_csv.csv'
data = pd.read_csv(url,sep=';')
data.head(5)



data.isnull().sum()

# Seleccion de las variables para nuestro conjunto X
#Se escogen las 11 primeras variables ya que la variable 12 es considerada nuestro fy, se crea un nuevo dataset X que contiene el conjunto de las 11 primeras variables.


X = data.iloc[:, 0:11]
X[:10]

# Se entrena el modelo
#Se entrena el modelo con un estado aletorio de 1 a 10 y un tamaño indicado de teste del 30% del dataset


x_train, x_test, y_train, y_test = train_test_split(X, data['quality'], test_size=0.3, random_state=int(np.random.randint(low=1, high=10, size=1)))



# Dos árboles de decisión, cada uno con un nivel diferente de profundidad.


# The decision tree classifier.
clf = tree.DecisionTreeClassifier(
    criterion= "entropy",
    splitter= "best",
    max_depth=2,
    #min_samples_split= 10,
    #min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    #min_impurity_decrease=0.05,
    class_weight=None,
    ccp_alpha = 0.0)
clf = clf.fit(x_train, y_train)

#tree.plot_tree(clf)
# esta es la función básica de sklearn para imprimir el grafo resultante

# la que viene a continuación es la versión de graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph

y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)

# The decision tree classifier.
clf = tree.DecisionTreeClassifier(
    criterion= "entropy",
    splitter= "best",
    max_depth=30,
    #min_samples_split= 10,
    #min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    #min_impurity_decrease=0.05,
    class_weight=None,
    ccp_alpha = 0.0)
clf = clf.fit(x_train, y_train)

#tree.plot_tree(clf)
# esta es la función básica de sklearn para imprimir el grafo resultante

# la que viene a continuación es la versión de graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph

y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)



#Dos árboles de decisión (diferentes a los anteriores), sin limitar la profundidad, pero cada uno con diferentes números de nodos hoja.


# The decision tree classifier.
clf = tree.DecisionTreeClassifier(
    criterion= "entropy",
    splitter= "best",
    # max_depth=3,
    #min_samples_split= 10,
    #min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=3,
    #min_impurity_decrease=0.05,
    class_weight=None,
    ccp_alpha = 0.0)
clf = clf.fit(x_train, y_train)

#tree.plot_tree(clf)
# esta es la función básica de sklearn para imprimir el grafo resultante

# la que viene a continuación es la versión de graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph

y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)

# The decision tree classifier.
clf = tree.DecisionTreeClassifier(
    criterion= "entropy",
    splitter= "best",
    # max_depth=3,
    #min_samples_split= 10,
    #min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=1000,
    #min_impurity_decrease=0.05,
    class_weight=None,
    ccp_alpha = 0.0)
clf = clf.fit(x_train, y_train)


# la que viene a continuación es la versión de graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph

y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)


# Dos árboles de decisión (diferentes a los anteriores), sin limitar la profundidad, pero cada uno con diferentes umbrales de ganancia al generar los nodos hijos.


# The decision tree classifier.
clf = tree.DecisionTreeClassifier(
    criterion= "entropy",
    splitter= "best",
    max_depth=None,
    #min_samples_split= 10,
    #min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0001,
    class_weight=None,
    ccp_alpha = 0.0)
clf = clf.fit(x_train, y_train)

#tree.plot_tree(clf)
# esta es la función básica de sklearn para imprimir el grafo resultante

# la que viene a continuación es la versión de graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph

y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)

# The decision tree classifier.
clf = tree.DecisionTreeClassifier(
    criterion= "entropy",
    splitter= "best",
    max_depth=None,
    #min_samples_split= 10,
    #min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.09,
    class_weight=None,
    ccp_alpha = 0.0)
clf = clf.fit(x_train, y_train)


# la que viene a continuación es la versión de graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph

y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)


#Dos árboles de decisión (diferentes a los anteriores), sin limitar la profundidad, pero podados con diferentes márgenes de confianza en la clasificación de los datos.


# The decision tree classifier.
clf = tree.DecisionTreeClassifier(
    criterion= "entropy",
    splitter= "best",
    max_depth=None,
    #min_samples_split= 10,
    #min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    #min_impurity_decrease=0.5,
    class_weight=None,
    ccp_alpha = 0.0)
clf = clf.fit(x_train, y_train)


# la que viene a continuación es la versión de graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph

y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)

# The decision tree classifier.
clf = tree.DecisionTreeClassifier(
    criterion= "entropy",
    splitter= "best",
    max_depth=None,
    #min_samples_split= 10,
    #min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    #min_impurity_decrease=0.5,
    class_weight=None,
    ccp_alpha = 0.2)
clf = clf.fit(x_train, y_train)


# la que viene a continuación es la versión de graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph

y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)


# Punto 2. Random Forest.
# Dos sistemas diferentes fijando el número de características que usados para la clasificación, pero variando el número de árboles.


clf = RandomForestClassifier(
          n_estimators =100, #cantidad de árboles
          #integer, optional (default=100)
          criterion="gini",
          #string, optional (default=”gini”)
          max_depth=None,
          #integer or None, optional (default=None)
          min_samples_split=2,
          #int, float, optional (default=2)
          min_samples_leaf=1,
          #int, float, optional (default=1)
          min_weight_fraction_leaf=0.0,
          #float, optional (default=0.)
          max_features="auto",# caracteristicas
          #int, float, string or None, optional (default=”auto”)
          max_leaf_nodes=None,
          #int or None, optional (default=None)
          min_impurity_decrease=0.0,
          #float, optional (default=0.)
          bootstrap=True,
          #boolean, optional (default=True)
          oob_score=False,
          #bool (default=False)
          n_jobs=None,
          #int or None, optional (default=None)
          random_state=None,
          #int, RandomState instance or None, optional (default=None)
          verbose=0,
          #int, optional (default=0)
          warm_start=False,
          #bool, optional (default=False)
          ccp_alpha=0.0,
          #non-negative float, optional (default=0.0)
          max_samples=None
          #int or float, default=None
          )

clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)

clf = RandomForestClassifier(
          n_estimators =10000, #cantidad de árboles
          #integer, optional (default=100)
          criterion="gini",
          #string, optional (default=”gini”)
          max_depth=None,
          #integer or None, optional (default=None)
          min_samples_split=2,
          #int, float, optional (default=2)
          min_samples_leaf=1,
          #int, float, optional (default=1)
          min_weight_fraction_leaf=0.0,
          #float, optional (default=0.)
          max_features="auto",# caracteristicas
          #int, float, string or None, optional (default=”auto”)
          max_leaf_nodes=None,
          #int or None, optional (default=None)
          min_impurity_decrease=0.0,
          #float, optional (default=0.)
          bootstrap=True,
          #boolean, optional (default=True)
          oob_score=False,
          #bool (default=False)
          n_jobs=None,
          #int or None, optional (default=None)
          random_state=None,
          #int, RandomState instance or None, optional (default=None)
          verbose=0,
          #int, optional (default=0)
          warm_start=False,
          #bool, optional (default=False)
          ccp_alpha=0.0,
          #non-negative float, optional (default=0.0)
          max_samples=None
          #int or float, default=None
          )

clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)

#Dos sistemas diferentes fijando el número de árboles, pero cambiando el número de características usado para la clasificación.


clf = RandomForestClassifier(
          n_estimators =100, #cantidad de árboles
          #integer, optional (default=100)
          criterion="gini",
          #string, optional (default=”gini”)
          max_depth=None,
          #integer or None, optional (default=None)
          min_samples_split=2,
          #int, float, optional (default=2)
          min_samples_leaf=1,
          #int, float, optional (default=1)
          min_weight_fraction_leaf=0.0,
          #float, optional (default=0.)
          max_features=0.01,# caracteristicas
          #int, float, string or None, optional (default=”auto”)
          max_leaf_nodes=None,
          #int or None, optional (default=None)
          min_impurity_decrease=0.0,
          #float, optional (default=0.)
          bootstrap=True,
          #boolean, optional (default=True)
          oob_score=False,
          #bool (default=False)
          n_jobs=None,
          #int or None, optional (default=None)
          random_state=None,
          #int, RandomState instance or None, optional (default=None)
          verbose=0,
          #int, optional (default=0)
          warm_start=False,
          #bool, optional (default=False)
          ccp_alpha=0.0,
          #non-negative float, optional (default=0.0)
          max_samples=None
          #int or float, default=None
          )

clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)

clf = RandomForestClassifier(
          n_estimators =100, #cantidad de árboles
          #integer, optional (default=100)
          criterion="gini",
          #string, optional (default=”gini”)
          max_depth=None,
          #integer or None, optional (default=None)
          min_samples_split=2,
          #int, float, optional (default=2)
          min_samples_leaf=1,
          #int, float, optional (default=1)
          min_weight_fraction_leaf=0.0,
          #float, optional (default=0.)
          max_features=10,# caracteristicas
          #int, float, string or None, optional (default=”auto”)
          max_leaf_nodes=None,
          #int or None, optional (default=None)
          min_impurity_decrease=0.0,
          #float, optional (default=0.)
          bootstrap=True,
          #boolean, optional (default=True)
          oob_score=False,
          #bool (default=False)
          n_jobs=None,
          #int or None, optional (default=None)
          random_state=None,
          #int, RandomState instance or None, optional (default=None)
          verbose=0,
          #int, optional (default=0)
          warm_start=False,
          #bool, optional (default=False)
          ccp_alpha=0.0,
          #non-negative float, optional (default=0.0)
          max_samples=None
          #int or float, default=None
          )

clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
CM = confusion_matrix(y_test, y_predict)
#[TP,FP;FN,TN]
print(CM)
score = clf.score(x_test, y_test)
print(score)


# Punto 3. Naïve Bayes Classifiers.

#Diseñe e implemente en Python un sistema de clasificación Bayesiano Ingenuo.


# Create a Naive Bayes object
nbf = GaussianNB()
#Split data into training and testing data
xf_train, xf_test, yf_train, yf_test, = train_test_split(X, data['quality'], test_size=0.3, random_state=1970)
#Training the model
nbf.fit(xf_train, yf_train)
Yf_pred = nbf.predict(xf_test)
print("Accuracy NB:",accuracy_score(yf_test, Yf_pred))

# Create a Naive Bayes object
nbf = GaussianNB()
#Split data into training and testing data
xf_train, xf_test, yf_train, yf_test, = train_test_split(X, data['quality'], test_size=0.99, random_state=1970)
#Training the model
nbf.fit(xf_train, yf_train)
Yf_pred = nbf.predict(xf_test)
print("Accuracy NB:",accuracy_score(yf_test, Yf_pred))
