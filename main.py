import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn import preprocessing

class RunMode: #
    train = 0
    test = 1

class NNet: #
    def __init__(self, features_number, clust_number):
        self.layers_number = 2 #Количество слоев
        layers = [features_number, clust_number] #обозначаем слои
        self.layers_weights = [np.random.rand(layers[0], layers[1])] #Веса
        self.layers_biases = [np.zeros(layers[1], dtype=float)] #Отклоенения

    A = 1.0
    def activation_function(self, x): #Функция активации которую он юзал
        return 1.0 / (1.0 + np.exp(-NNet.A*x))
    
    def delta(self, x): #Функция дельта с лекции
        return np.exp(-NNet.A*np.square(x*x))
    
# feauters * weight + bias - как производится отклонение
# функция активации добавляет нелинейности
    H = 0.001
    def run(self, features, run_mode, target = None): #Обучение нейросети
        for i in range(self.layers_number - 1):
            source_data = features # По чему мы кластеризуем
            features = np.dot(features, self.layers_weights[i]) + self.layers_biases[i] #наши данные готовые
            features = self.activation_function(features)
            if run_mode == RunMode.train: 
                v = target - features #весовая функция
                p = v * self.delta(v) * features * (1.0-features) #общая часть производной отклонеия и веса
                db = self.H * p.sum(axis = 0) #производная откл
                dw = self.H *  np.dot(source_data.transpose(), p)# производная веса
                self.layers_biases[i] += db #меняем уровни
                self.layers_weights[i] += dw
                print(v.sum())
        return features
def prepare_data(dataset_name, sep, targ_name):
    data = pd.read_csv(dataset_name, sep=sep)
    data = data.dropna()
    targ = pd.DataFrame(data[targ_name])
    features = data.drop([targ_name], axis = 1)
    features = (features-features.mean())/features.std()
    features = ((features - features.mean())/features.std()).to_numpy()
    ordinal_encoder = preprocessing.OrdinalEncoder(dtype=int)
    targ = pd.DataFrame(ordinal_encoder.fit_transform(targ), columns=targ.columns)
    points_number = len(features)
    cluster_number = len(targ.value_counts())
    features_number = features.shape[1]
    targ = np.array([np.array([int(targ[targ_name][n]==i) for i in range(cluster_number)]) for n in range(points_number)])
    return features, targ, points_number, cluster_number, features_number

features, targ, points_number, cluster_number, features_number = prepare_data('C:\\Users\\user\Documents\\University\\Data Mining\\lab8\\fake_bills.csv', ';', 'is_genuine')

net = NNet(features_number, cluster_number)
for _ in range(100):
    res = net.run(features, RunMode.train, targ)
    print(f"Точность - {round((res.argmax(axis = 1) == targ.argmax(axis = 1)).sum() / points_number * 100, 1)} %")

features, targ, points_number, cluster_number, features_number = prepare_data('C:\\Users\\user\Documents\\University\\Data Mining\\lab8\\students.csv', ',', 'UNS')

net = NNet(features_number, cluster_number)
for _ in range(300):
    res = net.run(features, RunMode.train, targ)
    print(f"Точность - {round((res.argmax(axis = 1) == targ.argmax(axis = 1)).sum() / points_number * 100, 1)} %")