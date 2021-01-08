
# ## Model


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate


import tensorflow
tensorflow.random.set_seed(2)

# ### Model 2: Concatenation Layer - LSTM and Past Value

feature_emb_dim = 768
feature_aud_dim = 29
batch_size = 8

# Text Encoder
# our text already is in the form of embeddings
input_text = keras.Input(shape=(max_len,feature_emb_dim))
# Add 1 bidirectional LSTM
layer1_t = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input_text)
output_t = layers.Dense(1, activation="tanh")(layer1_t)

# Audio Encoder
# our text already is in the form of embeddings
input_aud = keras.Input(shape=(max_len,feature_aud_dim))
# Add 1 bidirectional LSTM
layer1_a = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(input_aud)
output_a = layers.Dense(1, activation="tanh")(layer1_a)

# Fusion
fusion = concatenate([output_t, output_a])
fusion2 = layers.LSTM(1, return_sequences=True)(fusion)
fusion3 = layers.Reshape((fusion2.shape[1],))(fusion2)
fusion4 = layers.Dense(1, activation="linear")(fusion3)

layer_txt_aud = layers.Dense(64, activation="linear")(fusion4)
output_txt_aud = layers.Dense(1, activation="linear")(layer_txt_aud)

# Input from past val
input_past = keras.Input(shape=(1,))
input_final = concatenate([output_txt_aud, input_past])

# output volatility
output_vol = layers.Dense(1, activation="linear")(input_final)

# Input to the model includes text input, audio input and the past volatility
# Tying the entire model
model_lstm = Model(inputs=[input_text, input_aud,input_past], outputs=output_vol)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model_lstm.compile(loss='mean_squared_error', optimizer=opt)


# Train and test split
split= 200
train_txt = np.array(textData[:split])
train_aud = np.array(audioData[:split])
train_past = np.array(pre_vol[:split])

train_y = np.array(post_vol[:split])

test_txt = np.array(textData[split:])
test_aud = np.array(audioData[split:])
test_past = np.array(pre_vol[split:])

test_y = np.array(post_vol[split:])

# pre processing the data
# in this case we will have to preprocess each branch

# scaling the audio data
from sklearn.preprocessing import MinMaxScaler

scaler_layer = {}
for i in range(train_aud.shape[1]):
    scaler_layer[i] = StandardScaler()
    train_aud[:, i, :] = scaler_layer[i].fit_transform(train_aud[:, i, :]) 

for i in range(test_aud.shape[1]):
    test_aud[:, i, :] = scaler_layer[i].transform(test_aud[:, i, :]) 


# train the model
print("training model...")
model_lstm.fit(
x=[train_txt, train_aud,train_past], y=train_y,
epochs=25, batch_size=8)


results = model_lstm.evaluate([test_txt,test_aud,test_past], test_y, batch_size=8)
print("test loss, test acc:", results)




# from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# serialize model to JSON
model_json = model_lstm.to_json()
with open("model_lstm.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_lstm.save_weights("model_lstm.h5")
print("Saved model to disk")
 




# load json and create model
json_file = open('model_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_lstm.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer=opt,metrics=['mean_squared_error'])
score = loaded_model.evaluate([test_txt,test_aud,test_past],test_y, verbose=0)
print("%s: %.4f" % (loaded_model.metrics_names[1], score[1]))




print(loaded_model) 
print(loaded_model.summary())
# ### Model 1: Concatenation Layer - CNN and Past Values




feature_emb_dim = 768
feature_aud_dim = 29 # number of audio features
batch_size = 8

# Text Encoder
# our text already is in the form of embeddings
input_text = keras.Input(shape=(max_len,feature_emb_dim))
# Add 1 bidirectional LSTM
layer1_t = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input_text)
output_t = layers.Dense(1, activation="tanh")(layer1_t)

# Audio Encoder
# our audio is in the form of features
input_aud = keras.Input(shape=(max_len,feature_aud_dim))
# Add 1 bidirectional LSTM
layer1_a = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(input_aud)
output_a = layers.Dense(1, activation="tanh")(layer1_a)

# Fusion
fusion = concatenate([output_t, output_a])
fusion2 = layers.Reshape((max_len,2,1))(fusion)
fusion3 = layers.Conv2D(filters=1,kernel_size=(4,2),padding="valid",activation="linear",input_shape=(max_len,2))(fusion2)
fusion4 = layers.Reshape((fusion3.shape[1],))(fusion3)
fusion5 = layers.Dense(128, activation="linear")(fusion4)

layer_txt_aud = layers.Dense(64, activation="linear")(fusion5)

# combined output from both text and audio
output_txt_aud = layers.Dense(1, activation="linear")(layer_txt_aud)

# Input from past val
input_past = keras.Input(shape=(1,))
input_final = concatenate([output_txt_aud, input_past])

# output volatility
output_vol = layers.Dense(1, activation="linear")(input_final)
# Final Vector gives a dense layer as op


# Tying the entire model
model_cnn = Model(inputs=[input_text, input_aud,input_past], outputs=output_vol)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model_cnn.compile(loss='mean_squared_error', optimizer=opt)


# Train and test split
# Since this is dependent on time series, we cannot make use of random split

split= 200
train_txt = np.array(textData[:split])
train_aud = np.array(audioData[:split])
train_past = np.array(pre_vol[:split])

train_y = np.array(post_vol[:split])

test_txt = np.array(textData[split:])
test_aud = np.array(audioData[split:])
test_past = np.array(pre_vol[split:])

test_y = np.array(post_vol[split:])

# pre processing the data
# in this case we will have to preprocess each branch

# scaling the audio data
from sklearn.preprocessing import MinMaxScaler

scaler_layer = {}
for i in range(train_aud.shape[1]):
    scaler_layer[i] = StandardScaler()
    train_aud[:, i, :] = scaler_layer[i].fit_transform(train_aud[:, i, :]) 

for i in range(test_aud.shape[1]):
    test_aud[:, i, :] = scaler_layer[i].transform(test_aud[:, i, :]) 



# train the model
print("training model...")
model_cnn.fit(
x=[train_txt, train_aud,train_past], y=train_y,
epochs=25, batch_size=8)


results = model_cnn.evaluate([test_txt,test_aud,test_past], test_y, batch_size=8)
print("test loss, test acc:", results)

# to check predictions
preds = model_cnn.predict([test_txt,test_aud,test_past])
for i in range(len(preds)):
    print("{} {}".format(preds[i][0],test_y[i]))


# In[568]:


#!pip install h5py


# In[569]:


#!pip install keras


# In[552]:


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# serialize model to JSON
model_json = model_cnn.to_json()
with open("cnn3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_cnn.save_weights("cnn3.h5")
print("Saved model to disk")


# In[553]:


# load json and create model
json_file = open("cnn3.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("cnn3.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer=opt,metrics=['mean_squared_error'])
score = loaded_model.evaluate([test_txt,test_aud,test_past],test_y, verbose=0)
print("%s: %.4f" % (loaded_model.metrics_names[1], score[1]))


# In[554]:


test_txt.shape


