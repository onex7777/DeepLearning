# coding: utf-8
import numpy as np
import random
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, SGD, Adam


def sample(preds, temperature=1.0):
    eps = 1e-6
    preds = np.asarray(preds).astype('float64') + eps
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if __name__ == '__main__':
    f = open('三国演义.txt', encoding='utf-8')
    text = f.read()
    print(text[:300])
    f.close()
    print('Corpus length:', len(text))

    maxlen = 10
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('Number of sequences:', len(sentences))

    # List of unique characters in the corpus
    chars = sorted(list(set(text)))
    print('Unique characters:', len(chars))
    # Dictionary mapping unique characters to their index in `chars`
    char_indices = dict((char, chars.index(char)) for char in chars)
    print(char_indices)

    # Next, one-hot encode the characters into binary arrays.
    print('Vectorization...')
    print(len(sentences), maxlen, len(chars))
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # Building the network
    model = keras.models.Sequential()
    # model.add(layers.LSTM(64, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(layers.LSTM(64, return_sequences=False, input_shape=(maxlen, len(chars))))
    # model.add(layers.GRU(64, return_sequences=False))
    model.add(layers.Dense(len(chars), activation='softmax'))
    print('网络结构：', end='')
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
    for epoch in range(1, 60):
        print('epoch', epoch)
        model.fit(x, y, batch_size=128, epochs=1)
        # Select a text seed at random
        start_index = random.randint(0, len(text) - maxlen - 1)
        generated_text = text[start_index: start_index + maxlen]
        print('随机文本："' + generated_text + '"')

        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('------ temperature:', temperature)
            sys.stdout.write(generated_text)
            for i in range(400):    # 预测未来的400个字符
                sampled = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(generated_text):
                    sampled[0, t, char_indices[char]] = 1.
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]
                generated_text += next_char
                generated_text = generated_text[1:]
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
