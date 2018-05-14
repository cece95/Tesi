
# coding: utf-8

# # Sentence Embeddings
# ---
# https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle?mlreview

# In[1]:


import pandas as pd
import numpy as np

train = pd.read_csv('/usr/local/share/kaggle/Quora Question Pairs/uncompressed/train.csv')
test = pd.read_csv('/usr/local/share/kaggle/Quora Question Pairs/uncompressed/test.csv')


# ## Load Glove Dictionary

# In[2]:


# load the GloVe vectors in a dictionary:

embeddings_index = {}
with open('glove.840B.300d.txt', 'r') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found {} word vectors.'.format(len(embeddings_index)))


# ## Tokenize sentences

# In[ ]:


import keras.preprocessing.text as text
import keras.preprocessing.sequence as sequence

token = text.Tokenizer(num_words=None)
max_len = 70

#mi assicuro che tutti le frasi siano stringhe (rimuovo in NaN)

l1 = [str(x) for x in train['question1']]
l2 = [str(x) for x in train['question2']]

token.fit_on_texts(l1 + l2)
q1_seq = token.texts_to_sequences(l1)
q2_seq = token.texts_to_sequences(l2)

# zero pad the sequences
data1 = sequence.pad_sequences(q1_seq, maxlen=max_len)
data2 = sequence.pad_sequences(q2_seq, maxlen=max_len)

word_index = token.word_index


# ### fit on both training and test

# In[3]:


import keras.preprocessing.text as text
import keras.preprocessing.sequence as sequence

token = text.Tokenizer(num_words=None)
max_len = 70 #originally 70

#mi assicuro che tutti le frasi siano stringhe (rimuovo in NaN)
#training set
l1 = [str(x) for x in train['question1']]
l2 = [str(x) for x in train['question2']]
#test set
l1_t = [str(x) for x in test['question1']]
l2_t = [str(x) for x in test['question2']]

#token fit
token.fit_on_texts(l1 + l2 + l1_t + l2_t)

#train sequences
q1_seq = token.texts_to_sequences(l1)
q2_seq = token.texts_to_sequences(l2)
#test sequences
q1_seq_t = token.texts_to_sequences(l1_t)
q2_seq_t = token.texts_to_sequences(l2_t)

# zero pad the sequences
#train
data1 = sequence.pad_sequences(q1_seq, maxlen=max_len)
data2 = sequence.pad_sequences(q2_seq, maxlen=max_len)
#test
test1 = sequence.pad_sequences(q1_seq_t, maxlen=max_len)
test2 = sequence.pad_sequences(q2_seq_t, maxlen=max_len)

word_index = token.word_index


# ## Create Embedding Matrix

# In[4]:


embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
del embeddings_index


# ## Prepare data

# In[5]:


import keras.utils as kUtils

split = np.random.rand(len(data1)) < 0.8

y = train['is_duplicate']
y = kUtils.to_categorical(y, 2)

x1_train = data1[split]
x1_test = data1[~split]

x2_train = data2[split]
x2_test = data2[~split]

y_train = y[split]
y_test = y[~split]


# ## Simple LSTM 
# ---

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate

#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))


#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3)
lstm_layer = LSTM(100, dropout=0.3, recurrent_dropout=0.3)

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = lstm_layer(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = lstm_layer(x2)

final_input = concatenate([x1, x2], axis=1)
final_input = Flatten()(final_input)
output = Dense(1024, activation='relu')(final_input)
output = Dropout(0.2)(output)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(2, activation='sigmoid') (output)

model_bi = Model(inputs=[input_1, input_2], outputs=[output])
model_bi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_bi.fit([x1_train, x2_train], y_train, epochs=50, batch_size=32, validation_data=([x1_test, x2_test], y_test), verbose=0)
model_bi.save('my_model_emb.h5')


# ### Evaluate

# In[21]:


score = model.evaluate(x=[x1_test, x2_test], y=y_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# ## Bi-directional LSTM
# ---

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate, Bidirectional

#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))


#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3)
lstm_layer = Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = lstm_layer(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = lstm_layer(x2)

final_input = concatenate([x1, x2], axis=1)
output = Dense(1024, activation='relu')(final_input)
output = Dropout(0.2)(output)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(2, activation='sigmoid') (output)

model_bi = Model(inputs=[input_1, input_2], outputs=[output])
model_bi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_bi.fit([x1_train, x2_train], y_train, epochs=50, batch_size=32, validation_data=([x1_test, x2_test], y_test), verbose=0)
model_bi.save('my_model_emb_bi.h5')


# ### Evaluate

# In[24]:


score = model_bi.evaluate(x=[x1_test, x2_test], y=y_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# ## GRU
# ---

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate, GRU

#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))

#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3)
gru_layer_1 = GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)
gru_layer_2 = GRU(300, dropout=0.3, recurrent_dropout=0.3)

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = gru_layer_1(x1)
x1 = gru_layer_2(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = gru_layer_1(x2)
x2 = gru_layer_2(x2)

final_input = concatenate([x1, x2], axis=1)
output = Dense(1024, activation='relu')(final_input)
output = Dropout(0.2)(output)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(2, activation='sigmoid') (output)

model_gru = Model(inputs=[input_1, input_2], outputs=[output])
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_gru.fit([x1_train, x2_train], y_train, epochs=50, batch_size=32, validation_data=([x1_test, x2_test], y_test), verbose=0)
model_gru.save('my_model_emb_gru.h5')


# ### Evaluate

# In[30]:


score = model_gru.evaluate(x=[x1_test, x2_test], y=y_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# ## Doubled training
# ---

# In[38]:


x1_double_train = np.concatenate((x1_train, x2_train))
x2_double_train = np.concatenate((x2_train, x1_train))

x1_double_test = np.concatenate((x1_test, x2_test))
x2_double_test = np.concatenate((x2_test, x1_test))

y_double_train = np.concatenate((y_train, y_train))
y_double_test = np.concatenate((y_test, y_test))


# ## Simple LSTM

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate

#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))


#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3)
lstm_layer = LSTM(100, dropout=0.3, recurrent_dropout=0.3)

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = lstm_layer(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = lstm_layer(x2)

final_input = concatenate([x1, x2], axis=1)
output = Dense(1024, activation='relu')(final_input)
output = Dropout(0.2)(output)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(2, activation='sigmoid') (output)

model = Model(inputs=[input_1, input_2], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit([x1_double_train, x2_double_train], y_double_train, epochs=50, batch_size=32, validation_data=([x1_double_test, x2_double_test], y_double_test), verbose=0)
model.save('my_model_emb_double.h5')


# ### Evaluate

# In[41]:


score = model.evaluate(x=[x1_double_test, x2_double_test], y=y_double_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# # Predictions

# ## Train definitive model

# In[6]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate

#data doubled
#data1_double = np.concatenate((data1, data2))
#data2_double = np.concatenate((data2, data1))

#y_double = np.concatenate((y,y))

#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))


#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3) #increment dropout to 0.5
lstm_layer = LSTM(100, dropout=0.3, recurrent_dropout=0.3) #increment dropout to 0.5

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = lstm_layer(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = lstm_layer(x2)

final_input = concatenate([x1, x2], axis=1)
output = Dense(1024, activation='relu')(final_input)
output = Dropout(0.2)(output)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(2, activation='sigmoid') (output)

final_model = Model(inputs=[input_1, input_2], outputs=[output])
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

final_model.fit([data1, data2], y, epochs=10, batch_size=32, validation_split=0.1, verbose=2) #originally epoch: 50
final_model.save('definitive_model_fitBoth.h5')


# ## Simple LSTM  - modified
# * token size = 100
# * lstm output = 200

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate

#data doubled
#data1_double = np.concatenate((data1, data2))
#data2_double = np.concatenate((data2, data1))

#y_double = np.concatenate((y,y))

#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))

#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3) #increment dropout to 0.5
lstm_layer = LSTM(200, dropout=0.3, recurrent_dropout=0.3) #increment dropout to 0.5

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = lstm_layer(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = lstm_layer(x2)

final_input = concatenate([x1, x2], axis=1)
output = Dense(1024, activation='relu')(final_input)
output = Dropout(0.2)(output)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(2, activation='sigmoid') (output)

final_model = Model(inputs=[input_1, input_2], outputs=[output])
final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

final_model.fit([data1, data2], y, epochs=10, batch_size=32, validation_split=0.1, verbose=2) #originally epoch: 50
final_model.save('definitive_model_fitBoth_mod1.h5')


# In[ ]:


score = final_model.evaluate(x=[x1_test, x2_test], y=y_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# ### sigle token fit
# 
# * Test loss: 0.3593884353085545
# * Test accuracy: 0.8297356936778715
# 
# ### fit on both training and test - 10 epoch
# * Test loss: 0.38338456526880627
# * Test accuracy: 0.8196576333695905
# * Training time: 6h

# ## predict

# In[ ]:


predict = final_model.predict([test1, test2])


# In[ ]:


results = pd.DataFrame({
    'test_id': test['test_id'],
    'is_duplicate': predict[:,1]
})


# In[ ]:


results.to_csv('results_fitBoth_mod1.csv', index=False, header=True)


# ## Simple LSTM - cosine Similarity

# In[23]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate, Lambda
from keras import backend as K



#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))

#cosine similarity layer
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3)
lstm_layer = LSTM(100, dropout=0.3, recurrent_dropout=0.3)
dense_layer = Dense(100, activation='relu')

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = lstm_layer(x1)
x1 = dense_layer(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = lstm_layer(x2)
x2 = dense_layer(x2)

output = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([x1, x2])
output = Dense(2, activation='sigmoid') (output)

model = Model(inputs=[input_1, input_2], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=32, validation_data=([x1_test, x2_test], y_test), verbose=2)
model.save('my_model_emb_cosine_sim.h5')


# ### Evaluate

# In[25]:


score = model.evaluate(x=[x1_test, x2_test], y=y_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# ## V2

# In[29]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate, Lambda, Activation
from keras import backend as K



#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))

#cosine similarity layer
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3)
lstm_layer = LSTM(100, dropout=0.3, recurrent_dropout=0.3)
dense_layer = Dense(100, activation='relu')

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = lstm_layer(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = lstm_layer(x2)

output = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([x1, x2])
output = Activation(activation='sigmoid')(output)

model = Model(inputs=[input_1, input_2], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=32, validation_data=([x1_test, x2_test], y_test), verbose=2)
model.save('my_model_emb_cosine_sim_v2.h5')


# In[ ]:


# loss: 0.6918
# acc: 0.6309


# ## CONV1D

# In[19]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM, Dropout, concatenate, Flatten, Conv1D, MaxPooling1D
from keras import backend as K

#inputs
input_shape = max_len

input_1 = Input(shape=(input_shape,))
input_2 = Input(shape=(input_shape,))

#siamese layers
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
sp_dropout = SpatialDropout1D(0.3)

n = 3

l_conv1 = Conv1D(128, 5, activation='relu', padding='same')
l_pool1 = MaxPooling1D(5)
l_conv2 = Conv1D(128, 5, activation='relu', padding='same')
l_pool2 = MaxPooling1D(5)
l_conv3 = Conv1D(128, 5, activation='relu', padding='same')
#l_pool3 = MaxPooling1D(10) # global max pooling

# Architecture

x1 = embedding_layer(input_1)
x1 = sp_dropout(x1)
x1 = l_conv1(x1)
x1 = l_pool1(x1)
x1 = l_conv2(x1)
x1 = l_pool2(x1)
x1 = l_conv3(x1)
#x1 = l_pool3(x1)
x1 = Flatten()(x1)

x2 = embedding_layer(input_2)
x2 = sp_dropout(x2)
x2 = l_conv1(x2)
x2 = l_pool1(x2)
x2 = l_conv2(x2)
x2 = l_pool2(x2)
x2 = l_conv3(x2)
#x2 = l_pool3(x2)
x2 = Flatten()(x2)

final_input = concatenate([x1, x2], axis=1)
output = Dense(1024, activation='relu')(final_input)
output = Dropout(0.2)(output)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(2, activation='sigmoid') (output)

model = Model(inputs=[input_1, input_2], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# In[20]:


model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=32, validation_data=([x1_test, x2_test], y_test), verbose=2)
model.save('my_model_emb_conv1D.h5')


# ## CONV2D

# In[7]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, SpatialDropout1D, LSTM, Dropout, concatenate, Conv2D, Lambda, MaxPooling2D, Reshape


def layer_slice(i):
    def fun(x):
        return x[:,:,:,i]
    return Lambda(fun)

#inputs
input_shape = (max_len, )

input_1_conv = Input(shape=input_shape)
input_2_conv = Input(shape=input_shape)

#siamese layers
reshape = Reshape((max_len, 300, 1))
embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
slice_0 = layer_slice(0)
slice_1 = layer_slice(1)
conv_layer1 = Conv2D(2, (2, 300), activation='relu', padding='same')
conv_layer2 = Conv2D(2, (3, 300), activation='relu', padding='same')
conv_layer3 = Conv2D(2, (4, 300), activation='relu', padding='same')
pooling_layer = MaxPooling2D(pool_size=(2, 2), strides=None)
fc_layer = Dense(1024, activation='relu')

# Architecture

#q1
a = embedding_layer(input_1_conv)
a = reshape(a)
a1 = conv_layer1(a)
a2 = conv_layer2(a)
a3 = conv_layer3(a)

a11 = reshape(slice_0(a1))
a12 = reshape(slice_1(a1))
a21 = reshape(slice_0(a2))
a22 = reshape(slice_1(a2))
a31 = reshape(slice_0(a3))
a32 = reshape(slice_1(a3))

x1 = pooling_layer(a11)
x2 = pooling_layer(a12)
x3 = pooling_layer(a21)
x4 = pooling_layer(a22)
x5 = pooling_layer(a31)
x6 = pooling_layer(a32)

q1 = concatenate([x1,x2,x3,x4,x5,x6], axis=1)
q1 = Flatten()(q1)
q1 = fc_layer(q1)

#q2
b = embedding_layer(input_2_conv)
b = reshape(b)
b1 = conv_layer1(b)
b2 = conv_layer2(b)
b3 = conv_layer3(b)

b11 = reshape(slice_0(b1))
b12 = reshape(slice_1(b1))
b21 = reshape(slice_0(b2))
b22 = reshape(slice_1(b2))
b31 = reshape(slice_0(b3))
b32 = reshape(slice_1(b3))

y1 = pooling_layer(b11)
y2 = pooling_layer(b12)
y3 = pooling_layer(b21)
y4 = pooling_layer(b22)
y5 = pooling_layer(b31)
y6 = pooling_layer(b32)

q2 = concatenate([y1,y2,y3,y4,y5,y6], axis=1)
q2 = Flatten()(q2)
q2 = fc_layer(q2)

#shared layer
output = concatenate([q1, q2], axis=1)
#output = Flatten()(final_input)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(1024, activation='relu')(output)
output = Dropout(0.2)(output)
output_conv = Dense(2, activation='sigmoid') (output)

model_conv = Model(inputs=[input_1_conv, input_2_conv], outputs=[output_conv])
model_conv.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_conv.summary()


# In[ ]:


model_conv.fit([x1_train, x2_train], y_train, epochs=10, batch_size=32, validation_data=([x1_test, x2_test], y_test), verbose=2)
model.save('model_Conv2D_glove.h5')


# In[9]:


score = model_conv.evaluate(x=[x1_test, x2_test], y=y_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))

