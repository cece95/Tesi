
# coding: utf-8

# # Analisi del problema di Quora
# ---
# 
# Il problema è proposto da http://www.quora.com. Il sito propone un sistema per condividere le proprie conoscenze in vari ambiti e per espandere queste stesse conoscenze. Questo grazie ad un sistema di semplici domande e risposte, che sono però organizzate in vari topic (interessi) che le persone possono seguire per personalizzare la loro esperienza. E’ presente un profilo personale che mostra domande e risposte e le conoscenze principali di ogni persona.
# Il problema posto dall’azienda è il riconoscimento di domande che hanno lo stesso significato: sono molte infatti le domande che vengono ripetute (per pigrizia degli utenti o anche a causa della ricerca). Diventerebbe quindi importante ottimizzare il numero di domande unendo quelle uguali (con stesso scopo), ma anche (e soprattutto) unire le risposte, per rendere più completa possibile l’esperienza dell’utente riguardo lo specifico argomento della domanda.
# Attualmente il sistema utilizza un modello RandomForest per identificare le domande uguali: la sfida lanciata è quella di trovare degli algoritmi migliori di rilevamento usando tecniche avanzate di Machine Learning e di Natural Language Processing.
# 
# Il problema si presenta come una classificazione binaria con l'uso di addestramento supervisionato
# 
# # Analisi  Training Set
# ---
# 
# Abbiamo iniziato ad analizzare il dataset partendo dal Training Set (`train.csv`).
# L'header indica la denominazione delle 5 features e del label:
# 1. `id` = id della coppia di domande
# 2. `qid1` = id della prima domanda
# 3. `qid2` = id della seconda domanda
# 4. `question1` = testo della prima domanda
# 5. `question2` = testo della seconda domanda
# 6. `is_duplicate` = indica se le due domande sono uguali (è il label)

# In[1]:


get_ipython().run_cell_magic('time', '', "\nimport pandas as pd\nimport numpy as np\nimport time\n\n#Read dataset\ntrain = pd.read_csv('/usr/local/share/kaggle/Quora Question Pairs/uncompressed/train.csv')\n\n#Print dataframe columns types infos\nprint(train.dtypes)\n\n#Print some examples\ntrain.head()\nprint(len(train))")


# # Elaborazione dei dati
# ---
# 
# Inizialmente, seguendo i suggerimenti trovati in alcuni notebook presenti su kaggle, si era deciso di ripulire il dataset, eliminando tutto ciò che non fosse lettere o numeri. 
# Questa decisione si è rivelata poi controproducente in quanto l'algoritmo risulta meno performante su di un dataset "pulito".
# 
# Si noti la differenza tra una frase "sporca" e una "pulita"

# ### FASE 0A: Creazione dizionari
# 
# Inizialmente è utile costruire delle strutture dati che poi saranno impiegate nel corso dell'elaborazione dei dati; anche se esse vengono utilizzate in fasi diverse dell'elaborazione risulta utile crearle tutte insime in un unica operazione.
# Le strutture dati sono le seguenti
# 
# * `dictionary`: Dizionario che associa a ciascuna parola un id intero
# * `f_list`: lista che associa a ciascuna parola la propria frequenza, nello specifico contiene coppie (`id_parola`, `frequenza_parola`); la lista è ordinata in ordine decrescente di frequenza
# * `n_data`: dataset formato da due colonne contenente le coppie di domande, dove ciascuna domanda è rappresentata da un array di `id_parola`
# * `id_train`: `n_data` con l'aggiunta di una prima colonna contenente l'id della coppia di domande 
# 

# In[2]:


get_ipython().run_cell_magic('time', '', "\ndef build_dictionary(data):\n    #Initialization\n    dictionary = {} #dizionario: stringa -> id\n    c = 0\n    frequency_list = [] #lista frequenze: list[id] = (id_parola, frequenza_parola)\n    converted_data = [] #dataset che contiene numeri al posto delle parole \n    #Takes every word in every couple and creates a dictionary\n    for col in ['question1', 'question2']:\n        new_col = []\n        for sentence in data[col]:\n            sentence_converted = []\n            sentence = str(sentence).split(' ')\n            for p in sentence:\n                #Checks if word is already in dictionary\n                if str(p) not in dictionary.keys():\n                    #Checks length of word and if it's a space, then add\n                    if not (len(str(p)) == 1 and ord(str(p)) == 32):\n                        dictionary[str(p)] = c\n                        frequency_list.append((c,1))\n                        sentence_converted.append(c)\n                        c = c+1\n\n                else:\n                    c_old = dictionary[str(p)]\n                    (n, freq) = frequency_list[c_old]\n                    frequency_list[c_old] = (n, freq+1)\n                    sentence_converted.append(c_old)\n\n            new_col.append(sentence_converted)\n\n        converted_data.append(new_col)            \n\n    sorted_frequency_list = sorted(frequency_list, key=lambda pair: pair[1], reverse=True)                \n\n    return dictionary, sorted_frequency_list, converted_data\n\ndictionary, f_list, n_data = build_dictionary(train)\n\nid_train = pd.DataFrame({\n        'id': train['id'],\n        'question1': n_data[0],\n        'question2': n_data[1]\n        })")


# ### Fase 0B: Inizializzazione rete di test
# ---
# 
# La rete prescelta per la soluzione di questo problema è `MLPClassifier` presente all'interno del package `sklearn`.
# Come funzione di attivazione è stata scelta `logistic` in quanto la probabilità che le due domande siano un duplicato è rappresentata da un numero reale compreso tra 0 e 1 (come da istruzioni presenti sul sito)
# Come algoritmo solver è stato scelto `adam` perchè, dopo aver effettuato dei test di confronto con gli altri algoritmi, è risultato essere quello che dava il risultato migliore
# Come metrica di valutazione è stata scelta la funzione `LogLoss` in quanto è la stessa utilizzata da Kaggle
# 
# In questa fase seleziono anche il numero di elementi da usare come insieme di train e quanti invece come insieme di test

# In[3]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.losses import binary_crossentropy
import keras.utils as kUtils

#labels
duplicate = train['is_duplicate']

#initialize algorithm
alg = MLPClassifier(solver='adam', activation='logistic')

# split dataset train/test
split = np.random.rand(len(train)) < 0.8

#funzione di test
def test(features, k=0):
    
    features_train = features[split]
    features_test = features[~split]

    labels_train = duplicate[split]
    labels_test = duplicate[~split]

    alg.fit(features_train, labels_train)
    predict = alg.predict_proba(features_test)
    logLoss = log_loss(labels_test, predict)

    print('Sklearn Log Loss: {}'.format(logLoss))
    
    plot(predict[:,1], labels_test)
    
    ###KERAS NETWORK###
    features_train = np.asmatrix(features_train)
    features_test = np.asmatrix(features_test)
    
    labels_train = np.asmatrix(labels_train)
    labels_test = np.asmatrix(labels_test)
    
    labels_train = kUtils.to_categorical(labels_train, 2)
    labels_test = kUtils.to_categorical(labels_test, 2)
    
    model = Sequential()
    model.add(Dense(200, activation='relu', input_dim = features_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(features_train, labels_train,
          batch_size=128,
          epochs=20,
          verbose=0,
          validation_data=(features_test, labels_test))
    
    score = model.evaluate(features_test, labels_test, verbose=0)
    keras_predict = model.predict_proba(features_test)

    print('Keras Log loss: {}'.format(score[0]))
    #plot(keras_predict[:,1], labels_test)
    
    ####################
    
    if (k == 1):
        return predict[:,1]
    
    

import matplotlib.pyplot as plt    

#funzione per mostrare la distribuzione delle predizioni rispetto alle label di test
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


# ### FASE 0C: Multithreading
# ---
# 
# Scrivo una funzione che suddivida il lavoro su 4 core in modo da velocizzare l'esecuzione dell'algoritmo

# In[4]:


from threading import Thread
import numpy as np

def split_work(fun, train):    
    dataframes = np.array_split(train, 4)

    threads = []
    results = {}
    i = 0
    
    for data in dataframes:
        t = Thread(target=fun, args=(data, results, i))
        threads.append(t)
        t.start()
        i = i+1

    for t in threads:
        t.join()

    features = []    
    for key in sorted(results):
        features.extend(results[key])
    
    return features


# ### FASE 0D: Salvataggio e caricamento delle strutture dati
# ---

# In[5]:


def save(s, o):
    obj = np.asarray(o)
    np.save(s, obj)
        
def load(s):
    return np.load(s)

def compute(fun, data, file):
    if not os.path.exists(file):
        res = split_work(fun, data)
        save(file, res)
    else:
        res = load(file)
    
    return res


# ### FASE 1: Estrazione Features di base
# ---
# 
# Come prima cosa risulta utile estrarre le feature più semplici che possono caratterizzare una frase:
# 
# * `q1_words`: numero di parole contenute nella prima domanda
# * `q1_letters`: numero di caratteri contenuti nella prima domanda
# * `q2_words`: numero di parole contenute nella seconda domanda
# * `q2_letters`: numero di lettere contenute nella seconda domanda
# 
# Infatti risulta piuttosto intuitivo che frasi di simile lunghezza hanno più probabilità di avere lo stesso significato

# In[6]:


get_ipython().run_cell_magic('time', '', "import os\n\ndef base_extraction(data):\n\n    if not os.path.exists('base_features.csv'):\n        #Count number of words and letters for each 'question1'\n        q1_words = data['question1'].apply(lambda x: len(str(x).split(' ')))\n        q1_letters = data['question1'].apply(lambda x: len(str(x)))\n\n        #Count number of words and letters for each 'question2'\n        q2_words = data['question2'].apply(lambda x: len(str(x).split(' ')))\n        q2_letters = data['question2'].apply(lambda x: len(str(x)))\n\n        features = pd.DataFrame({\n                    'q1_words': q1_words,\n                    'q1_letters': q1_letters,\n                    'q2_words': q2_words,\n                    'q2_letters': q2_letters,\n                    })\n        \n        features.to_csv('base_features.csv', index=False, header=list(features))\n    else:\n        features = pd.read_csv('base_features.csv')\n    \n    return features\n\nfeatures = base_extraction(train)\n\n#TEST\n\n#test(features)")


# ### Fase 2: LCS
# ---
# 
# Dovendo stabilire la similarità tra due frasi un buon indicatore può essere la presenza di sottosequenze comuni, ho quindi deciso di prendere come parametro la lunghezza della più lunga sottostringa comune tra le due domande.

# In[7]:


get_ipython().run_cell_magic('time', '', "\ndef LCSLength(data, results, k):\n    res =[]\n    \n    for row in data.itertuples():\n        X = getattr(row, 'question1')\n        Y = getattr(row, 'question2')\n        m = len(X)\n        n = len(Y)\n        C = np.zeros((m,n), dtype=np.int8);\n        for i in range(m):\n            for j in range(n):\n                if X[i] == Y[j]:\n                    C[i,j] = C[i-1,j-1] + 1\n                else:\n                    C[i,j] = max(C[i,j-1], C[i-1,j])    \n        res.append(C[m-1,n-1])\n    results[k] = res\n\nLCSs = compute(LCSLength, id_train, 'LCS.npy')\nfeatures = features.assign(**{'LCS' : LCSs})\n\n#TEST\n\n#test(features)")


# ### Fase 3: Word Similar Share
# ---
# 
# Appoggiandomi alla libreria `Spacy`, libreria specifica per il Natural Language Processing, ho calcolato un indice di similarità tra le due domande come somma degli indici di similarità tra ciascuna coppia di parole (`w1`,`w2`) dove `w1` appartiene alla prima domanda e `w2` alla seconda
# 
# Inizialmente avevo normalizzato l'indice rispetto alla lunghezza delle due domande ma la predizione risultava più precisa senza normalizzazione

# In[8]:


import spacy
nlp = spacy.load('en_core_web_md')


# In[9]:


def word_similar_share(data, results, i):
    res = []
    for row in data.itertuples():   
        q1 = getattr(row, 'question1')
        q2 = getattr(row, 'question2')
    
        q1words = nlp(str(q1))
        q2words = nlp(str(q2))

        jac = 0

        for w1 in q1words:
            s_w1 = str(w1)
            if not (len(s_w1) == 1 and ord(s_w1) == 32):
                for w2 in q2words:
                    s_w2 = str(w2)
                    if not (len(s_w2) == 1 and ord(s_w2) == 32):
                        jac = jac + w1.similarity(w2)
        
        res.append(jac)
    results[i] = res
############################################

#word_match = split_work(word_similar_share, train)    
#features_tmp = features.assign(**{'Word_Match' : word_match})

#TEST

#test(features_tmp)


# ### Fase 4: Common/Rare Words
# ---
# 
# In miglioramento dell'algoritmo visto alla fase 3 è l'introduzione del concetto di parola comune e parola rara
# Una parola è considerata rara se la sua frequenza è bassa, e similmente è considerata comune se la sua frequenza è alta
# 
# Si ottengono quindi due algoritmi distinti, entrambi ignorano le parole comuni
# * `word_match_rare_function`: prendendo in considerazione le parole rare si ottiene un indice di similarità che viene incrementato di 1 per ogni parola rara che le due domande hanno in comune e decrementato di 1 per ogni parola rara che compare in una sola delle due domande
# * `word_similar_share_cr`: come il `word_similar_share` visto al punto 3
# 
# Anche in questo caso normalizzare gli indici ne riduce la capacità predittiva

# In[10]:


def word_match_rare_function(data, results, i):
    res = []
    for row in data.itertuples():   
        q1words = getattr(row, 'question1')
        q2words = getattr(row, 'question2')

        rare_jac = 0
        #n_rare = 0
        q1_rares = set()
        q2_rares = set()

        for w1 in q1words:
            if w1 in rare_words:
                q1_rares.add(w1)

        for w2 in q2words:
            if w2 in rare_words:
                q2_rares.add(w2)

        for w1r in q1_rares:
            if w1r in q2_rares:
                rare_jac = rare_jac + 1
            else:
                rare_jac = rare_jac - 1

        for w2r in q2_rares:
            if not w2r in q1_rares:
                rare_jac = rare_jac - 1
        '''                
        try:
            rare_jac = rare_jac/n_rare
        except:
            rare_jac = 0                            
        '''
        res.append(rare_jac)
    results[i] = res

def word_similar_share_cr(data, results, i):
    res = []
    for row in data.itertuples():   
        q1 = getattr(row, 'question1')
        q2 = getattr(row, 'question2')
    
        q1words = nlp(str(q1))
        q2words = nlp(str(q2))

        jac = 0

        for w1 in q1words:
            s_w1 = str(w1)
            if not (len(s_w1) == 1 and ord(s_w1) == 32 and s_w1 not in common_words):
                for w2 in q2words:
                    s_w2 = str(w2)
                    if not (len(s_w2) == 1 and ord(s_w2) == 32 and s_w2 not in common_words):
                        jac = jac + w1.similarity(w2)
                try:
                    jac = jac/(len(q1words)+len(q2words))
                except:
                    jac = 0                
        res.append(jac)
    results[i] = res

common_words = set()
rare_words = set()

#build rare words set
for (p,f) in f_list:
    if f<=20:
        rare_words.add(p)

#Build common words set
for (p,f) in f_list[:10]:
    common_words.add(p)

#Find similarity with Jaccard for each question pair
#word_match_cr = split_work(word_similar_share_cr, train)
#word_match_rare = split_work(word_match_rare_function, id_train)

#features_tmp = features.assign(**{'Word_Match_cr' : word_match_cr})
#features_tmp = features.assign(**{'Word_Match_rare' : word_match_rare})

#TEST

#test(features_tmp)


# ### Fase 4B: vari valori per Common e Rare
# ---
# 
# I valori di frequenza per le parole comuni e rare al punto 4 erano stati scelti arbitrariamente, quindi in questa fase si testeranno vari valori per tali frequenze per determinare le migliori

# In[11]:


# start = time.time()

# for F in range(20, 110, 10):
#     for n in range(4, 11):
#         data_tmp = features.copy()
#         C = pow(2,n)

#         common_words = set()
#         rare_words = set()

#         #build rare words list
#         for (p,f) in f_list:
#             if f<=F:
#                 rare_words.add(p)

#         #Build common words list
#         for (p,f) in f_list[:C]:
#             common_words.add(p)

#         #Find similarity with Jaccard for each question pair
#         word_match_cr = train.apply(word_similar_share_cr, axis=1, raw=True)
#         word_match_rare = id_train.apply(word_match_rare_function, axis=1, raw=True)

#         data_tmp = data_tmp.assign(**{'Word_Match_cr' : word_match_cr})
#         data_tmp = data_tmp.assign(**{'Word_Match_rare' : word_match_rare})

#         #TEST

#         features_train = features_tmp[split]
#         features_test = features_tmp[~split]

#         labels_train = duplicate[split]
#         labels_test = duplicate[~split]

#         alg.fit(features_train, labels_train)
#         predict = alg.predict_proba(features_test)
#         logLoss = log_loss(labels_test, predict)

#         results = []
#         results.append((C,F,logLoss))
#         print("Log loss C: {}, F: {} = {}".format(C,F,logLoss))

# sorted_results = sorted(results, key=lambda pair: pair[2])
# print(sorted_results[:5])
# end = time.time()
# print('time: {}'.format(end - start))


# ### Fase 4C: deduzioni
# ---
# 
# Dall'esecuzione della fase 5 si deduce che i parametri che danno il risultato migliore sono `C = 64 ` e `F = 50`. Non è detto che si rivelino buoni parametri anche per il set di test ma dato che comunque l'algoritmo basato sulle parole rare si ha dimostrato di dare buoni risultati terrò questi parametri validi anche per il caso di test.
# In ogni caso il valore dei parametri incide poco sul risultato finale, in quanto il valore migliore risulta essere `0.4996397287043904` mentre il peggiore è `0.5068634596789862` per una differenza di `0.00722373097`

# In[12]:


get_ipython().run_cell_magic('time', '', "\nF = 50\nC = 64\n\ndef word_match_rare_function(data, results, i):\n    res = []\n    for row in data.itertuples():   \n        q1words = getattr(row, 'question1')\n        q2words = getattr(row, 'question2')\n\n        rare_jac = 0\n        #n_rare = 0\n        q1_rares = set()\n        q2_rares = set()\n\n        for w1 in q1words:\n            if w1 in rare_words:\n                q1_rares.add(w1)\n\n        for w2 in q2words:\n            if w2 in rare_words:\n                q2_rares.add(w2)\n\n        for w1r in q1_rares:\n            if w1r in q2_rares:\n                rare_jac = rare_jac + 1\n            else:\n                rare_jac = rare_jac - 1\n\n        for w2r in q2_rares:\n            if not w2r in q1_rares:\n                rare_jac = rare_jac - 1\n        '''                \n        try:\n            rare_jac = rare_jac/n_rare\n        except:\n            rare_jac = 0                            \n        '''\n        res.append(rare_jac)\n    results[i] = res\n\ndef word_similar_share_cr(data, results, i):\n    res = []\n    for row in data.itertuples():   \n        q1 = getattr(row, 'question1')\n        q2 = getattr(row, 'question2')\n    \n        q1words = nlp(str(q1))\n        q2words = nlp(str(q2))\n\n        jac = 0\n\n        for w1 in q1words:\n            s_w1 = str(w1)\n            if not (len(s_w1) == 1 and ord(s_w1) == 32 and s_w1 not in common_words):\n                for w2 in q2words:\n                    s_w2 = str(w2)\n                    if not (len(s_w2) == 1 and ord(s_w2) == 32 and s_w2 not in common_words):\n                        jac = jac + w1.similarity(w2)\n                '''\n                try:\n                    jac = jac/(len(q1words)+len(q2words))\n                except:\n                    jac = 0\n                '''    \n        res.append(jac)\n    results[i] = res\n\ncommon_words = set()\nrare_words = set()\n\n#build rare words set\nfor (p,f) in f_list:\n    if f<=F:\n        rare_words.add(p)\n\n#Build common words set\nfor (p,f) in f_list[:C]:\n    common_words.add(p)\n\n#Find similarity with Jaccard for each question pair\nword_match_cr = compute(word_similar_share_cr, train, 'word_match_cr.npy')\nword_match_rare = compute(word_match_rare_function, id_train, 'word_match_rare.npy')\n    \nfeatures = features.assign(**{'Word_Match_cr' : word_match_cr})\nfeatures = features.assign(**{'Word_Match_rare' : word_match_rare})\n\n#TEST\n\n#test(features)")


# ### Fase 5: Cosine Similarity
# ---
# 
# Un altro utile parametro per la similarità tra frasi è la Cosine similarity: una tecnica euristica per la misurazione della similitudine tra due vettori effettuata calcolando il coseno tra di loro.
# 
# Per poterla applicare è quindi necessario convertire ciascuna frase in una `Bag of words`, cioè un'array di dimensione pari al numero di parole distinte presenti all'interno del dataset. In questo array la posizione i-esima è settata a 1 se la parola comprare nella frase, 0 altrimenti.
# In pratica ciascuna frase è rappresentata come un vettore in uno spazio n-dimensionale (con n pari alla dimensione di ciascuna `Bag of words`)
# A questo punto si applica semplicemente la cosine similarity tra i due vettori ottenuti

# In[13]:


# start = time.time()

# #bag words size
# dim = len(dictionary.keys())
# print("bag words size: {}".format(dim))

# #converte ogni frase in una bag of words da usare per la cosine similarity
# def cosineConversion(sentence):
#     s = np.zeros(dim)
#     for w in sentence:
#         s[w] = 1
#     return s
    
# #simple wrap for cosineConversion
# def cosineConversion_wrap(data):
#     c1 = []
#     c2 = []
#     for sent1 in data['question1']:
#         s1 = cosineConversion(sent1)
#         c1.append(s1)

#     for sent2 in data['question2']:
#         s2 = cosineConversion(sent2)
#         c2.append(s2)

#     return c1, c2

# #calculate Cosine Similarity
# def cosineSimilarity(row):
#     a = row['question1']
#     b = row['question2']

#     cos_sim = dot(a, b)/(norm(a)*norm(b))

#     return cos_sim



# cosine1, cosine2 = cosineConversion_wrap(id_train)	

# cosine_data = pd.DataFrame({
#             'id': train['id'],
#             'question1': cosine1,
#             'question2': cosine2
#             })

# mid = time.time()

# print('Cosine Similarity dataset: {}'.format(mid - start))

# cosine_similarity = cosine_data.apply(cosineSimilarity, axis=1, raw=True)

# features_tmp = features.copy().assign(**{'Cosine Similarity' : cosine_similarity})

# end = time.time()
# print('Cosine Similarity Algorithm: {}'.format(end - mid))

# #TEST

# test(features_tmp)


# Sfortunatamente questa prima versione dell'algoritmo occupa troppa memoria 

# ### Fase 5B: cosine similarity migliorata
# ---
# Questa versione dell'algoritmo è più efficiente rispetto alla precedente e occupa molta meno memoria. Inoltre invece che far assumere a ciascun elemento della `Bag of words` i valori 0/1, `Sentence[w]` varrà  `n` cioè il numero di apparizioni della data parola all'interno della frase

# In[14]:


from numpy.linalg import norm

#bag words size
dim = len(dictionary.keys())

def cosineConversion(sentence):
    s = np.zeros(dim, dtype=np.int8)
    for w in sentence:
        s[w] = s[w] + 1
    return s

def cosineSimilarity(data, results, i):
    res = []
    for row in data.itertuples():   
        q1 = getattr(row, 'question1')
        q2 = getattr(row, 'question2')
    
        a = cosineConversion(q1)
        b = cosineConversion(q2)

        cos_sim = np.dot(a, b)/(norm(a)*norm(b))

        res.append(cos_sim)
    results[i] = res

    
#cosine_similarity = split_work(cosineSimilarity, id_train)
#features_tmp = features.assign(**{'Cosine Similarity' : cosine_similarity})

#TEST

#test(features_tmp)


# ### Fase 5C: Cosine Similarity Sklearn
# ---
# 
# La `Cosine Similarity` sembrava un buon parametro ma l'algoritmo precedentemente utilizzato non dava il risultato sperato, così mi sono appoggiato all'algoritmo presente all'interno del pacchetto `Sklearn`

# In[15]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import TfidfVectorizer as TF\nfrom sklearn.metrics.pairwise import cosine_similarity\n\ndef cosineSimilarity(data, results, i):\n    res = []\n    for row in data.itertuples():   \n        q1 = str(getattr(row, 'question1'))\n        q2 = str(getattr(row, 'question2'))\n\n        vect = TF(min_df=1)\n        tfidf = vect.fit_transform([q1, q2])\n\n        res.append((tfidf * tfidf.T).A[0,1])\n    results[i] = res\n    \ndef cosineSimilarity2(data):\n    vect = TF(min_df=1)\n    l1 = data['question1']\n    l2 = data['question2']\n    res = []\n    \n    vect.fit(data['question1'] + data['question2'])\n    for row in data.itertuples():   \n        q1 = str(getattr(row, 'question1'))\n        q2 = str(getattr(row, 'question2'))\n\n        tf1 = vect.transform(q1)\n        tf2 = vect.transform(q2)\n\n        cs = cosine_similarity(tf1,tf2)\n        res.append(cs)\n\n#cos_sim = compute(cosineSimilarity, train, 'cosine_similarity.npy')\ncos_sim = cosineSimilarity2(train)\nfeatures = features.assign(**{'cos_sim': cos_sim})\n\n#TEST\n#test(features)")


# La cosine similarity calcolata da Sklearn risulta essere migliore rispetto a quella calcolata precedente

# ### Fase 6: Spacy Sentence similarity
# ---
# 
# La libreria Spacy prevede la possibilità di calcolare direttamente un indice di similarità tra frasi, questo si va ad aggiungere ai parametri precedenti 

# In[16]:


get_ipython().run_cell_magic('time', '', "\ndef sentenceSimilarity(data, results, i):\n    res = []\n    for row in data.itertuples():   \n        q1s = getattr(row, 'question1')\n        q2s = getattr(row, 'question2')\n    \n        q1 = nlp(str(q1s))\n        q2 = nlp(str(q2s))\n\n        res.append(q1.similarity(q2))\n    results[i] = res\n\nspacy_sim = compute(sentenceSimilarity, train, 'spacy_similarity.npy')\nfeatures = features.assign(**{'spacy_sim': spacy_sim})     \n\n#TEST\n\n#test(features)\n\nfeatures_set = features")


# ## Recap
# ---
# 
# I parametri finora individuati sono:
# * features di base
# * LCS
# * jaccard common/rare
# * cosine similarity (Sklearn)
# * Spacy sentence similarity
# 
# essi risultano ottimi per l'individuazione di frasi non duplicate, risulta invece più difficile individuare i doppioni

# ## GridSearch

# In[18]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
'''
X_train = features[split]
X_test = features[~split]

y_train = duplicate[split]
y_test = duplicate[~split]

possible_parameters = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam']
}

alg = MLPClassifier()
loss_f = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

# The GridSearchCV is itself a classifier
# we fit the GridSearchCV with the training data
# and then we use it to predict on the test set
clf = GridSearchCV(alg, possible_parameters, n_jobs=4, scoring=loss_f) # n_jobs=4 means we parallelize the search over 4 threads
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)
logLoss = log_loss(y_test, y_pred)

logLoss
'''


# In[ ]:


'''
cv = clf.cv_results_
tab = pd.DataFrame({
    'mean_fit_time': cv['mean_fit_time'],
    'mean_score_time': cv['mean_score_time'],
    'mean_test_score': cv['mean_test_score'],
    'mean_train_score': cv['mean_train_score'],
    'param_activation': cv['param_activation'],
    'param_solver': cv['param_solver'],
    'rank_test_score': cv['rank_test_score']
})

tab.sort_values(['rank_test_score']).head(3)
'''


# In[26]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.model_selection import GridSearchCV\nfrom sklearn.metrics import make_scorer\n\nalg = MLPClassifier(solver='adam', activation='logistic')\nloss_f = make_scorer(log_loss, greater_is_better=False, needs_proba=True)\n\nX_train = features[split]\nX_test = features[~split]\n\ny_train = duplicate[split]\ny_test = duplicate[~split]\n\npossible_parameters = {\n    'alpha': [0.0001, 0.00005],\n    'batch_size': [100, 50],\n    'learning_rate_init': [0.001, 0.0005],\n    'beta_1': [0.9, 0.99],\n    'beta_2': [0.999, 0.5],\n    'epsilon': [5e-9, 2e-9]\n}\n\nclf2 = GridSearchCV(alg, possible_parameters, n_jobs=4, scoring=loss_f) # n_jobs=4 means we parallelize the search over 4 threads\nclf2.fit(X_train, y_train)\n\ny_pred = clf2.predict_proba(X_test)\nlogLoss = log_loss(y_test, y_pred)\n\nlogLoss")


# In[28]:


cv = clf2.cv_results_
tab = pd.DataFrame({
    'mean_fit_time': cv['mean_fit_time'],
    'mean_score_time': cv['mean_score_time'],
    'mean_test_score': cv['mean_test_score'],
    'mean_train_score': cv['mean_train_score'],
    'param_alpha': cv['param_alpha'],
    'param_batch_size': cv['param_batch_size'],
    'param_learning_rate_init': cv['param_learning_rate_init'],
    'rank_test_score': cv['rank_test_score'],
    'param_beta_1': cv['param_beta_1'],
    'param_epsilon': cv['param_epsilon']
})

tab.sort_values(['rank_test_score']).head(3)


# ## Random Forest Classifier

# In[39]:


get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestClassifier\n\nrf = RandomForestClassifier(random_state=0, n_jobs=4)\n\npossible_parameters = {\n    'max_depth': [x for x in range(1, 20, 1)],\n    'criterion': ['gini', 'entropy'],\n    'max_features': ['auto', 'log2', None]\n}\n\nclf = GridSearchCV(rf, possible_parameters, n_jobs=4, scoring=loss_f)\n\nclf.fit(X_train, y_train)")


# In[42]:


y_pred = clf.predict_proba(X_test)
logLoss = log_loss(y_test, y_pred)

logLoss


# Max_depth = 2, gini: 0.5456

# In[41]:


cv_rf = clf.cv_results_
tab = pd.DataFrame({
    'mean_fit_time': cv_rf['mean_fit_time'],
    'mean_score_time': cv_rf['mean_score_time'],
    'mean_test_score': cv_rf['mean_test_score'],
    'mean_train_score': cv_rf['mean_train_score'],
    'param_max_depth': cv_rf['param_max_depth'],
    'param_criterion': cv_rf['param_criterion'],
    'param_max_features': cv_rf['param_max_features'],
    'rank_test_score': cv_rf['rank_test_score'],
})

tab.sort_values(['rank_test_score']).head(3)


# ## Normalize Data

# In[22]:


from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

features_scaled = pd.DataFrame(columns = list(features))

letters_scaler = MinMaxScaler()
words_scaler = MinMaxScaler()

letters_scaler.fit(features['q1_letters'] + features['q2_letters'])
features_scaled['q1_letters'] = letters_scaler.transform(features['q1_letters'])
features_scaled['q2_letters'] = letters_scaler.transform(features['q2_letters'])

words_scaler.fit(features['q1_words'] + features['q2_words'])
features_scaled['q1_words'] = words_scaler.transform(features['q1_words'])
features_scaled['q2_words'] = words_scaler.transform(features['q2_words'])

for feat in  ['LCS', 'Word_Match_cr', 'Word_Match_rare', 'cos_sim', 'spacy_sim']:
    features_scaled[feat] = MinMaxScaler().fit_transform(features[feat])


# ### MLPClassifier

# In[24]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

alg = MLPClassifier(solver='adam', activation='logistic')
loss_f = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

X_train = features_scaled[split]
X_test = features_scaled[~split]

y_train = duplicate[split]
y_test = duplicate[~split]

possible_parameters = {
    'alpha': [0.0001, 0.00005],
    'batch_size': [100, 50],
    'learning_rate_init': [0.001, 0.0005],
    'beta_1': [0.9, 0.99],
    'beta_2': [0.999, 0.5],
    'epsilon': [5e-9, 2e-9]
}

mlp = GridSearchCV(alg, possible_parameters, n_jobs=4, scoring=loss_f) # n_jobs=4 means we parallelize the search over 4 threads
mlp.fit(X_train, y_train)


# In[25]:


cv = mlp.cv_results_
tab = pd.DataFrame({
    'mean_fit_time': cv['mean_fit_time'],
    'mean_score_time': cv['mean_score_time'],
    'mean_test_score': cv['mean_test_score'],
    'mean_train_score': cv['mean_train_score'],
    'param_alpha': cv['param_alpha'],
    'param_batch_size': cv['param_batch_size'],
    'param_learning_rate_init': cv['param_learning_rate_init'],
    'rank_test_score': cv['rank_test_score'],
    'param_beta_1': cv['param_beta_1'],
    'param_epsilon': cv['param_epsilon']
})

tab.sort_values(['rank_test_score']).head(3)


# ### Random Forest

# In[29]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0, n_jobs=4)

possible_parameters = {
    'max_depth': [x for x in range(1, 20, 1)],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'log2', None]
}

possible_parameters['max_depth'].append(None)

clf = GridSearchCV(rf, possible_parameters, n_jobs=4, scoring=loss_f)

clf.fit(X_train, y_train)


# In[30]:


cv_rf = clf.cv_results_
tab = pd.DataFrame({
    'mean_fit_time': cv_rf['mean_fit_time'],
    'mean_score_time': cv_rf['mean_score_time'],
    'mean_test_score': cv_rf['mean_test_score'],
    'mean_train_score': cv_rf['mean_train_score'],
    'param_max_depth': cv_rf['param_max_depth'],
    'param_criterion': cv_rf['param_criterion'],
    'param_max_features': cv_rf['param_max_features'],
    'rank_test_score': cv_rf['rank_test_score'],
})

tab.sort_values(['rank_test_score']).head(3)


# In[21]:


X_train = features[split]
X_test = features[~split]

y_train = duplicate[split]
y_test = duplicate[~split]


# ### Naive Bayes Gaussian

# In[33]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict_proba(X_test)
logLoss = log_loss(y_test, y_pred)

logLoss


# ### Quadratic Discriminant

# In[34]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

y_pred = qda.predict_proba(X_test)
logLoss = log_loss(y_test, y_pred)

logLoss


# ### Support Vector Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn import svm\n\nsvc = svm.SVC()\nsvc.fit(X_train, y_train)\n\ny_pred = svc.predict_proba(X_test)\nlogLoss = log_loss(y_test, y_pred)\n\nlogLoss')


# ## FC Network

# In[17]:


get_ipython().run_cell_magic('time', '', "\nfeatures_train = features[split]\nfeatures_test = features[~split]\n\nlabels_train = duplicate[split]\nlabels_test = duplicate[~split]\n\nfeatures_train = np.asmatrix(features_train)\nfeatures_test = np.asmatrix(features_test)\n\nlabels_train = np.asmatrix(labels_train)\nlabels_test = np.asmatrix(labels_test)\n\nlabels_train = kUtils.to_categorical(labels_train, 2)\nlabels_test = kUtils.to_categorical(labels_test, 2)\n\n#architecture\nmodel = Sequential()\nmodel.add(Dense(300, activation= 'elu', input_dim = features_train.shape[1]))\n#model.add(Dropout(0.2))\nmodel.add(BatchNormalization())\nfor i in range(8):\n    model.add(Dense(200, activation= 'elu'))\n    #model.add(Dropout(0.2))\n    #model.add(BatchNormalization())\nmodel.add(Dense(2, activation= 'softmax'))\n\nmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n\nhistory= model.fit(features_train, labels_train, batch_size=128, epochs=20, verbose=2, validation_data=(features_test, labels_test))\n\nscore = model.evaluate(features_test, labels_test, verbose=0)\n\nprint('Log loss: {}'.format(score[0]))\nprint('Accuracy: {}'.format(score[1]))")


# ### Add Dropout and Normalization

# In[19]:


get_ipython().run_cell_magic('time', '', "\nfeatures_train = features[split]\nfeatures_test = features[~split]\n\nlabels_train = duplicate[split]\nlabels_test = duplicate[~split]\n\nfeatures_train = np.asmatrix(features_train)\nfeatures_test = np.asmatrix(features_test)\n\nlabels_train = np.asmatrix(labels_train)\nlabels_test = np.asmatrix(labels_test)\n\nlabels_train = kUtils.to_categorical(labels_train, 2)\nlabels_test = kUtils.to_categorical(labels_test, 2)\n\n#architecture\nmodel = Sequential()\nmodel.add(Dense(300, activation= 'elu', input_dim = features_train.shape[1]))\nmodel.add(Dropout(0.2))\nmodel.add(BatchNormalization())\nfor i in range(8):\n    model.add(Dense(200, activation= 'elu'))\n    model.add(Dropout(0.2))\n    model.add(BatchNormalization())\nmodel.add(Dense(2, activation= 'softmax'))\n\nmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n\nhistory= model.fit(features_train, labels_train, batch_size=128, epochs=20, verbose=2, validation_data=(features_test, labels_test))\n\nscore = model.evaluate(features_test, labels_test, verbose=0)\n\nprint('Log loss: {}'.format(score[0]))\nprint('Accuracy: {}'.format(score[1]))")


# ### Esperimenti

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfeatures_train = features[split]\nfeatures_test = features[~split]\n\nlabels_train = duplicate[split]\nlabels_test = duplicate[~split]\n\nfeatures_train = np.asmatrix(features_train)\nfeatures_test = np.asmatrix(features_test)\n\nlabels_train = np.asmatrix(labels_train)\nlabels_test = np.asmatrix(labels_test)\n\nlabels_train = kUtils.to_categorical(labels_train, 2)\nlabels_test = kUtils.to_categorical(labels_test, 2)\n\n#architecture\nmodel = Sequential()\nmodel.add(Dense(400, activation= 'elu', input_dim = features_train.shape[1]))\nfor i in range(13):\n    model.add(Dense(400, activation= 'elu'))\nmodel.add(Dense(2, activation= 'softmax'))\n\nmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n\nhistory= model.fit(features_train, labels_train, batch_size=128, epochs=20, verbose=2, validation_data=(features_test, labels_test))\n\nscore = model.evaluate(features_test, labels_test, verbose=0)\n\nprint('Log loss: {}'.format(score[0]))\nprint('Accuracy: {}'.format(score[1]))")

