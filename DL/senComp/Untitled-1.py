# %%
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# %%
df = pd.read_csv("qoute_dataset.csv")

# %%
df.head()

# %%
quotes = df["quote"]
quotes.head()

# %%
quotes[0]

# %%
quotes = quotes.str.lower()

# %%
quotes[0]

# %%
import string

translator = str.maketrans("", "", string.punctuation)
quotes = quotes.apply(lambda x: x.translate(translator))

# %%
quotes[0]

# %%
from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 10000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(quotes)

# %%
word_index = tokenizer.word_index
print(len(word_index))

# %%
sentences = tokenizer.texts_to_sequences(quotes)

# %%
X = []
y = []

for seq in sentences:
    for i in range(1, len(seq)):
        input_seq = seq[:i]
        output_seq = seq[i]
        X.append(input_seq)
        y.append(output_seq)


# %%
len(X)

# %%
len(y)

# %%
max_len = max(len(x) for x in X)
print(max_len)

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_pad = pad_sequences(X, maxlen=max_len, padding="pre")

# %%
X_pad

# %%
y = np.array(y)

# %%
X_pad.shape

# %%
from tensorflow.keras.utils import to_categorical

y_onehot = to_categorical(y, num_classes=vocab_size)

# %%
y_onehot.shape

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# %%
embedded_dim = 50
rnn_units = 128


# %%
lstm_model = Sequential()
lstm_model.add(
    Embedding(input_dim=vocab_size, output_dim=embedded_dim, input_length=max_len)
)
lstm_model.add(LSTM(units=rnn_units))
lstm_model.add(Dense(units=vocab_size, activation="softmax"))

# %%
lstm_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# %%
lstm_model.summary()

# %%
epochs = 100
batch_size = 128


# %%
lstm_model.fit(X_pad, y_onehot, epochs=epochs, batch_size=batch_size, verbose=1)

# %%
from tensorflow.keras.models import load_model

lstm_model = load_model("lstm_model.h5")

# %%
lstm_model.save("lstm_model.h5")

# %%
index_to_word = {}
for word, index in word_index.items():
    index_to_word[index] = word


# %%
def predictor(model, text, tokenizer, max_len):

    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")

    pred = model.predict(seq, verbose=0)
    pred_index = np.argmax(pred)
    return index_to_word[pred_index]


# %%
seed_text = "what are you"
next_word = predictor(lstm_model, seed_text, tokenizer, max_len)
print(next_word)


# %%
def generate_text(model, seed_text, num_words, tokenizer, max_len):
    text = seed_text
    for i in range(num_words):
        next_word = predictor(model, text, tokenizer, max_len)
        if next_word == "":
            break
        text += " " + next_word
    return text


# %%
seed_text = "life is a"
generate_text = generate_text(lstm_model, seed_text, 10, tokenizer, max_len)
print(generate_text)

# %%
import pickle

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# %%
with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)

# %%
