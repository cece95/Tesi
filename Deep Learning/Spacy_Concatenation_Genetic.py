
# coding: utf-8

# # Spacy Vector Concatenation - Genetic Algorithm

# In[1]:


import os
import numpy as np
from sklearn.preprocessing import StandardScaler

#load data
data1 = np.load('data1.npy')
data2 = np.load('data2.npy')
duplicate = np.load('duplicate.npy')
data_a = np.concatenate([data1, data2], axis=1)
data_b = np.concatenate([data2, data1], axis=1)

data = np.concatenate([data_a, data_b], axis=0) 
y = np.concatenate([duplicate, duplicate], axis=0) 

del data1
del data2

#preprocessing
split = np.random.rand(len(data)) < 0.8

scl = StandardScaler()

data_scl = scl.fit_transform(data)

#features_train = data_scl[split]
#features_test = data_scl[~split]

#labels_train = duplicate[split]
#labels_test = duplicate[~split]


# In[2]:


y.shape


# In[3]:


#SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_scl, y, test_size=0.2, random_state=42)
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=0)


# In[19]:


#NETWORK
from numpy import random
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, BatchNormalization
from keras.losses import binary_crossentropy

class Network():

    def __init__(self):
        self.accuracy = 0
        self.params = {}

    def createRandom(self):
        self.params['layers'] = random.randint(1,6)
        self.params['neurons'] = random.choice([100, 200, 400, 500, 1000])
        self.params['activation'] = random.choice(['relu', 'elu', 'softplus'])
        self.params['output_activation'] = random.choice(['sigmoid', 'softmax'])
        self.params['dropout'] = random.randint(1,5)/10

    def printSelf(self):
        print('layers: {}\nneurons: {}\nactivation: {}\noutput_activation: {}\n dropout: {}'
              .format(self.params['layers'], self.params['neurons'], self.params['activation'], self.params['output_activation'], self.params['dropout']))    

    def buildNetwork(self, inputShape):
        model = Sequential()

        for i in range(self.params['layers']):
            if i == 0:
                model.add(Dense(self.params['neurons'], activation=self.params['activation'], input_shape=inputShape))
                model.add(Dropout(self.params['dropout']))
                model.add(BatchNormalization())
            else:
                model.add(Dense(self.params['neurons'], activation=self.params['activation']))
                model.add(Dropout(self.params['dropout']))
                model.add(BatchNormalization())

        model.add(Dense(2, activation=self.params['output_activation']))

        #BINARY CROSSENTROPY perchè è una classificazione binaria
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model;

    def breed(self,partner):
        child = Network()

        p1 = random.choice(['neurons', 'layers', 'activation', 'output_activation', 'dropout'])
        p2 = random.choice(['neurons', 'layers', 'activation', 'output_activation', 'dropout'])
        while (p1 == p2):
            p1 = random.choice(['neurons', 'layers', 'activation', 'output_activation', 'dropout'])
            p2 = random.choice(['neurons', 'layers', 'activation', 'output_activation', 'dropout'])

        child.params[p1] = self.params[p1]
        child.params[p2] = self.params[p2]
        
        remaining = [x for x in ['neurons', 'layers', 'activation', 'output_activation', 'dropout'] if x not in [p1, p2]]
        
        for p in remaining:
            child.params[p] = partner.params[p]

        return child

    def train_evaluate(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=0,
              validation_data=(x_test, y_test))
        print('\n')

        self.accuracy = self.model.evaluate(x_test, y_test, verbose=0)[0]
        
    def mutate(self):
        p1 = random.choice(['neurons', 'layers', 'activation', 'output_activation', 'dropout'])
        if p1 == 'neurons':
            self.params[p1] = random.choice([50, 100, 200, 400, 500, 1000])
        elif p1 == 'layers':
            self.params[p1] = random.randint(1,10)
        elif p1 == 'activation':
            self.params[p1] = random.choice(['relu', 'elu', 'softplus'])
        elif p1 == 'output_activation':
            self.params[p1] = random.choice(['sigmoid', 'softmax'])
        elif p1 == 'dropout':
            self.params[p1] = random.randint(1,5)/10


# In[20]:


population_results = []
population_results_val = []

#CREATE POPULATION
def createInitialPopulation(n):
    print('creating initial population')
    pop = [];
    for i in range(n):
        print('create network {}'.format(i+1))
        pop.append(Network())
        pop[i].createRandom()
        pop[i].printSelf()
        print('\n')
    return pop;

#FITNESS
def fitness(pop, x_train, y_train, x_test, y_test, input_shape, saved):
    i = 1
    sumScore = 0
    for net in pop:
        print('Fitness Rete {}'.format(i))
        if i > saved:
            net.buildNetwork(input_shape)
            print('Training\n')
            net.train_evaluate(x_train, y_train, x_test, y_test)
        else:
            print('Saved\n')
        sumScore = sumScore + 1 - net.accuracy;
        i = i + 1
    return sumScore	

#EVOLVE
#Manca da aggiungere una mutazione casuale
def evolve(pop, score):
    prob = []
    for net in pop:
        prob.append((1-net.accuracy)/score)
    
    tot = np.sum(prob)
    if tot > 1:
        prob[-1] = prob[-1] - (tot - 1)
    elif tot < 1:
        prob[0] = prob[0] + (1 - tot)
    print(tot)

    newPop = []
    newPop.extend([pop[0], pop[1]])
    for _ in range(len(pop)-4):
        g1 = random.choice(np.arange(0, 20), p=prob)
        g2 = random.choice(np.arange(0, 20), p=prob)
        while (g1 == g2):
            g1 = random.choice(np.arange(0, 20), p=prob)
            g2 = random.choice(np.arange(0, 20), p=prob)
        newPop.append(pop[g1].breed(pop[g2]))

    #mutation
    id1 = random.randint(0,20)
    id2 = random.randint(0,20)
    m1 = pop[id1]
    m1.mutate()
    m2 = pop[id2]
    m2.mutate()
    
    newPop.extend([m1,m2])
    return newPop


# In[21]:


#TEST
import keras.utils as kUtils

def genetic_test(x_train, x_test, y_train, y_test, x_val, y_val):
    generations = 10
    n_elements = 20

    y_train = kUtils.to_categorical(y_train, 2) #2 dovuto al fatto che è una classificazione binaria
    y_test = kUtils.to_categorical(y_test, 2)
    y_val = kUtils.to_categorical(y_val, 2)

    input_shape = (x_train.shape[1],)

    #ALGORITHM
    population = createInitialPopulation(n_elements)
    pop_score = fitness(population, x_train, y_train, x_test, y_test, input_shape, 0)
    population.sort(key=lambda x: x.accuracy)
    scores = [x.accuracy for x in population]
    print(scores)
    population_results.append(population[0].model.evaluate(x_val, y_val, verbose=0)[0])
    population_results_val.append(population[0].model.evaluate(x_test, y_test, verbose=0)[0])
    for i in range(generations-1):
        print('Generazione {}'.format(i+2))
        population = evolve(population, pop_score)
        pop_score = fitness(population, x_train, y_train, x_test, y_test, input_shape, 2)
        population.sort(key=lambda x: x.accuracy)
        scores = [x.accuracy for x in population]
        print(scores)
        population_results.append(population[0].model.evaluate(x_val, y_val, verbose=0)[0])
        population_results_val.append(population[0].model.evaluate(x_test, y_test, verbose=0)[0])

    final_alg = population[0]
    
    return final_alg


# In[22]:


result = genetic_test(X_train, X_test, y_train, y_test, X_val, y_val)


# In[23]:


import matplotlib.pyplot as plt

plt.plot(population_results)
plt.title('model Log Loss')
plt.ylabel('Log Loss')
plt.xlabel('Generation')
plt.show()


# In[24]:


import matplotlib.pyplot as plt

plt.plot(population_results_val)
plt.title('model Log Loss')
plt.ylabel('Log Loss')
plt.xlabel('Generation')
plt.show()


# In[26]:


result.printSelf()


# In[4]:


from numpy import random
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, BatchNormalization
from keras.losses import binary_crossentropy
import keras.utils as kUtils

model = Sequential()
inputShape = (X_train.shape[1],)

y_train = kUtils.to_categorical(y_train, 2)
y_test = kUtils.to_categorical(y_test, 2)

for i in range(3):
    if i == 0:
        model.add(Dense(1000, activation='relu', input_shape=inputShape))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
    else:
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

model.add(Dense(2, activation='softmax'))

#BINARY CROSSENTROPY perchè è una classificazione binaria
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train,
              batch_size=128,
              epochs=10,
              verbose=2,
              validation_data=(X_test, y_test))


# In[ ]:


import matplotlib.pyplot as plt
predict = model.predict(X_test)

def plot(p, labels_test):

    t = []
    f = []

    for p, l in zip(p, labels_test):
        if l == True:
            t.append(p)
        else:
            f.append(p)
        
    plt.hist(f, bins=20, normed=True, label='Not Duplicate')
    plt.hist(t, bins=20, normed=True, alpha=0.7, label='Duplicate')
    plt.show()
    
plot(predict, y_test[:,1])


# In[7]:


# -*- coding: utf-8 -*-
"""Spacy Convent
Descrizione:
    Applicazione di una rete convoluzionale al problema di Quora, basandomi sul word embedding
    fornito da spacy
"""
import numpy as np
import spacy
import pandas as pd
import time
#import dask.dataframe as dd
import os

#SPACY

nlp = spacy.load('en_core_web_md')

"""Features extraction
Descrizione:
    Funzione usata per fare l'embedding delle frasi, restituisce due matrici distinte, una per la 
    colonna 'question1' e una per la colonna 'question2'

Argomenti:
    'data': dataset da processare

Return:
    'r1': matrice contenente la colonna 'question1' processata
    'r2': matrice contenente la colonna 'question2' processata 

TODO:
    verificare se in questo caso può essere utile usare la pipe di spacy per velocizzare la 
    computazione
"""
def features_extraction(data):
    c1 = []
    c2 = []

    for row in data.itertuples():
        q1 = getattr(row, 'question1')
        q2 = getattr(row, 'question2')

        s1 = nlp(str(q1))
        c1.append(s1.vector)
        s2 = nlp(str(q2))
        c2.append(s2.vector)

    r1 = np.asmatrix(c1)
    r2 = np.asmatrix(c2)
    return r1, r2


# In[8]:


from keras.models import load_model
from sklearn.preprocessing import StandardScaler

test = pd.read_csv('/usr/local/share/kaggle/Quora Question Pairs/uncompressed/test.csv')
test1, test2 = features_extraction(test)

data_test = np.concatenate([test1, test2], axis=1) 

del test1
del test2

#preprocessing
scl = StandardScaler()

test_scl = scl.fit_transform(data_test)


# In[9]:


predict = model.predict(test_scl)

results = pd.DataFrame({
    'test_id': test['test_id'],
    'is_duplicate': predict[:,1]
})

results.to_csv('results_mlp_double.csv', index=False, header=True)

