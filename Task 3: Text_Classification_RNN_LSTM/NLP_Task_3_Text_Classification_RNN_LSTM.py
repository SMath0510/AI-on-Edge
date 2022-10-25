{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "Loading the embeddings from Glove dataset."
      ],
      "metadata": {
        "id": "4YFS95HAYERW"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fA_huZ7xdJJC"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Downloading the Glove embedding\n",
        "!wget http://nlp.stanford.edu/data/glove.6B.zip\n",
        "!unzip glove.6B.zip\n",
        "!ls -lat"
      ],
      "metadata": {
        "id": "727z9giovdEK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# There are 4 options - 50,100,200,300 dimensional embeddings\n",
        "# Let's choose the 50 dimensional one for our use\n",
        "\n",
        "vocab,embeddings = [],[]\n",
        "with open('glove.6B.50d.txt','rt') as fi:\n",
        "    full_content = fi.read().strip().split('\\n')\n",
        "for i in range(len(full_content)):\n",
        "    i_word = full_content[i].split(' ')[0]\n",
        "    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]\n",
        "    vocab.append(i_word)\n",
        "    embeddings.append(i_embeddings)"
      ],
      "metadata": {
        "id": "bVaxhtaswY74"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Below, is an example of how word is mapped to a 50 dimensional embeddings\n",
        "# Both vocab and embeddings are lists.\n",
        "for i in range (0,5):\n",
        "  print(vocab[i], \" - \", embeddings[i])"
      ],
      "metadata": {
        "id": "ue7HB9b5wkio"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Text - Classification (Sentiment Analysis)"
      ],
      "metadata": {
        "id": "LofY_o4BX823"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
      ],
      "metadata": {
        "id": "3z3-TPfcYUTN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "vocab_npa = np.array(vocab)\n",
        "embs_npa = np.array(embeddings)"
      ],
      "metadata": {
        "id": "_yKrV2WZyPby"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#insert '<pad>' and '<unk>' tokens at start of vocab_npa.\n",
        "vocab_npa = np.insert(vocab_npa, 0, '<pad>')\n",
        "vocab_npa = np.insert(vocab_npa, 1, '<unk>')\n",
        "print(vocab_npa[:10])\n",
        "\n",
        "pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.\n",
        "unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.\n",
        "\n",
        "#insert embeddings for pad and unk tokens at top of embs_npa.\n",
        "embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))\n",
        "print(embs_npa.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MQupJxUJyI62",
        "outputId": "53c2b4c2-dc3f-4683-cbb7-77be59fa942c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['<pad>' '<unk>' 'the' ',' '.' 'of' 'to' 'and' 'in' 'a']\n",
            "(400002, 50)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())\n",
        "\n",
        "assert my_embedding_layer.weight.shape == embs_npa.shape\n",
        "print(my_embedding_layer.weight.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vQ3HZXDtyGoB",
        "outputId": "faddfb99-679b-4bb5-b4bb-23b716eacdda"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "torch.Size([400002, 50])\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "with open('vocab_npa.npy','wb') as f:\n",
        "    np.save(f,vocab_npa)\n",
        "\n",
        "with open('embs_npa.npy','wb') as f:\n",
        "    np.save(f,embs_npa)"
      ],
      "metadata": {
        "id": "nVbagBISyV4p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from keras.datasets import imdb\n",
        "from keras import Sequential\n",
        "from keras.layers import LSTM, Embedding, Dense\n",
        "from keras.preprocessing.sequence import pad_sequences\n",
        "# fix random seed for reproducibility\n",
        "np.random.seed(7)\n",
        "# load the dataset but only keep the top 6000 words\n",
        "(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=6000)\n",
        "# pad input sequences\n",
        "X_train = pad_sequences(X_train, maxlen=500)\n",
        "X_test = pad_sequences(X_test, maxlen=500)\n",
        "#model\n",
        "model = Sequential()\n",
        "model.add(Embedding(6000, 32, input_length=500))\n",
        "model.add(LSTM(100))\n",
        "model.add(Dense(1, activation='sigmoid'))\n",
        "model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
        "print(model.summary())\n",
        "# Final evaluation of the model\n",
        "scores = model.evaluate(X_test, y_test, verbose=0)\n",
        "print(\"Accuracy: %.2f%%\" % (scores[1]*100))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Gc31_ODwE2i1",
        "outputId": "9c8a9a27-e016-46e8-fde9-f4cd610dc899"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential_2\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " embedding (Embedding)       (None, 500, 32)           192000    \n",
            "                                                                 \n",
            " lstm (LSTM)                 (None, 100)               53200     \n",
            "                                                                 \n",
            " dense (Dense)               (None, 1)                 101       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 245,301\n",
            "Trainable params: 245,301\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "None\n",
            "Accuracy: 50.09%\n"
          ]
        }
      ]
    }
  ]
}
