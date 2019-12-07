###################### Parte I -   Importacao das Bibliotecas #####################


import os
import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score



TAMANHO_ST=720

##################### Parte II -  Preprocessamento #####################

raiz = os.path.curdir

print("Abrindo os dados brutos em CSV")

condition_group_path = os.path.join(raiz, "./depresjon-dataset/condition/condition_{}.csv")
control_group_path = os.path.join(raiz, "./depresjon-dataset/control/control_{}.csv")

condition_raw_data = [
    np.array(pd.read_csv(condition_group_path.format(x), usecols=['activity']))
    for x in range(1, 24)
]

control_raw_data = [
    np.array(pd.read_csv(control_group_path.format(x), usecols=['activity']))
    for x in range(1, 32)
]

print("Truncando os dados = ", TAMANHO_ST)
#Truncando os dados em janelas de 720min = 12horas
truncate = lambda series: series[:TAMANHO_ST * (len(series) // TAMANHO_ST)]

condition_data = np.concatenate(
    [truncate(series).reshape(-1, TAMANHO_ST) for series in condition_raw_data])
control_data = np.concatenate(
    [truncate(series).reshape(-1, TAMANHO_ST) for series in control_raw_data])


#Normalizando dados
print("Normalizando os dados...")
sc = StandardScaler()
condition_data_transformed = sc.fit_transform(condition_data)
control_data_transformed = sc.fit_transform(control_data)


print("Concatenando os dados Condition e Control...")
X = np.concatenate((condition_data_transformed, control_data_transformed), axis=0)
y = (np.array([1] * len(condition_data_transformed) + [0] * len(control_data_transformed)))

# Dividindo conjunto de dados de treino e teste, 80/20,
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= np.random.randint(1, 2**16))

##################### Parte III -  Construindo a Rede #####################

print("Iniciando a ANN...")
# Inicializando a ANN
#parametros:
batch=32
epochs=50
cross_val=10


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn = build_classifier, batch_size = batch, epochs = epochs)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = cross_val, n_jobs = -1)
classifier.fit(X_train, y_train)
mean = accuracies.mean()
variance = accuracies.std()

y_pred = classifier.predict(X_test)

print("TREINO Resultados:")
print("Média de acurácia: ", round(mean,2))
print("Variância: ", round(variance,4))
print("Parâmetros: ", "\nbatch_size:", batch, "\tepochs:", epochs, "\tcv: ", cross_val)



# Matriz de confusão - teste
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\n\nTESTE Resultados:")
Precision = cm[0][0] / (cm[0][0]+cm[1][0])
Accuracy = (cm[0][0]+cm[1][1])/ (cm[0][0]+cm[1][0]+cm[1][1]+cm[0][1])
Recall = cm[0][0] / (cm[0][0]+cm[0][1])
print("Precision: ", round(Precision, 2))
print("Recall: ", round(Recall, 2))
print("Accuracy: ", round(Accuracy, 2))



################################ Parte IV OPT - Seleção de melhores parametros ##########################


#Otimizacao dos os parametros, melhores: adam, epoch=50, batch=32, cv=10
#OBS Utilizado para encontrar os melhores parametros da rede

# Testa as combinacoes dos parametros (batch, epochs e optmizer) e retorna o melhor baseado no score de acuracia
# com 10-fold cross validacao
#parametros a serem testados, diferentes tamanhos de batch, de epochs e optmizer
#classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size': [16, 32],
#              'epochs': [50, 100],
#              'optmizer': ['adam', 'rmsprop']}
#grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring= 'accuracy', cv=10)

#grid_search2 = grid_search.fit(X_train, y_train)
# retorna os melhores parametros para os dados utilizados
#best_parameters = grid_search2.best_params_
# retorna a melhor acuracia obtida dessa selecao
#best_accuracy = grid_search2.best_score_

#print("Resultados Obtidos: ")
#print("Melhor acurácia: ",best_accuracy)
#print("Melhores parâmetros: ", best_parameters)