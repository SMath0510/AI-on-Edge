import numpy as np
import torch
import torch.nn as nn

# Downloading the Glove embedding
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
!ls -lat

# There are 4 options - 50,100,200,300 dimensional embeddings
# Let's choose the 50 dimensional one for our use

vocab,embeddings = [],[]
with open('glove.6B.50d.txt','rt') as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)

# Below, is an example of how word is mapped to a 50 dimensional embeddings
# Both vocab and embeddings are lists.
for i in range (0,5):
  print(vocab[i], " - ", embeddings[i])

"""# Text - Classification (Sentiment Analysis)"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)

#insert '<pad>' and '<unk>' tokens at start of vocab_npa.
vocab_npa = np.insert(vocab_npa, 0, '<pad>')
vocab_npa = np.insert(vocab_npa, 1, '<unk>')
print(vocab_npa[:10])

pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

#insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
print(embs_npa.shape)

my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())

assert my_embedding_layer.weight.shape == embs_npa.shape
print(my_embedding_layer.weight.shape)

with open('vocab_npa.npy','wb') as f:
    np.save(f,vocab_npa)

with open('embs_npa.npy','wb') as f:
    np.save(f,embs_npa)

import numpy as np
from keras.datasets import imdb
from keras import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing.sequence import pad_sequences
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top 6000 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=6000)
# pad input sequences
X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)
#model
model = Sequential()
model.add(Embedding(6000, 32, input_length=500))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))