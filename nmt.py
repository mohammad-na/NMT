# ______________ embedding_new ______________________
import pandas as pd
import numpy as np
import string
from string import digits
import re
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Flatten
import sklearn
from nltk.translate.bleu_score import corpus_bleu
from matplotlib import pyplot
import psutil
import keras
import keras.backend as K
from keras.objectives import cosine_proximity
import numpy as np

# ............................................................................................#
# this function remove noise from word 
def deNoise(text):
    noise = re.compile(""" ّ    | 
                                                  َ    | 
                                                  ً    | 
                                                  ُ    |
                                                  ٌ    |
                                                  ِ    |
                                                  ٍ    |
                                                  ْ    | 
                                                  ـ     
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text


# ..................Vocalizing English Data...........................
lines = pd.read_table('dataset.txt', names=['eng', 'ar'])

lines = sklearn.utils.shuffle(lines)
lines = lines.reset_index(drop=True)

lines = lines[0:10000]
print(lines.shape)
# cleanup
lines.eng = lines.eng.apply(lambda x: x.lower())
lines.ar = lines.ar.apply(lambda x: x.lower())
lines.ar = lines.ar.apply(lambda x: deNoise(x))

lines.eng = lines.eng.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
lines.ar = lines.ar.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))

exclude = set(string.punctuation)
lines.eng = lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines.ar = lines.ar.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

remove_digits = str.maketrans('', '', digits)
lines.eng = lines.eng.apply(lambda x: x.translate(remove_digits))
lines.ar = lines.ar.apply(lambda x: x.translate(remove_digits))

print(lines.head())

lines.ar = lines.ar.apply(lambda x: 'START_ ' + x + ' _END')

all_eng_words = set()
for eng in lines.eng:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_arabic_words = set()
for ar in lines.ar:
    for word in ar.split():
        if word not in all_arabic_words:
            all_arabic_words.add(word)

print(len(all_eng_words), len(all_arabic_words))

lenght_list = []
for l in lines.ar:
    lenght_list.append(len(l.split(' ')))
max_ar = np.max(lenght_list)
print(max_ar)

lenght_list = []
for l in lines.eng:
    lenght_list.append(len(l.split(' ')))
max_en = np.max(lenght_list)
print(max_en)

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_arabic_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_arabic_words)


input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])

# __________________Embedding_____________________
MAX_input_LENGTH = max_en
MAX_output_LENGTH = max_ar
MAX_NUM_WORDS = 200
en_EMBEDDING_DIM = 300
ar_EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.3
En_vocab_size = len(all_eng_words) + 1
Ar_vocab_size = len(all_arabic_words) + 1
# ______ English embedding ___________
f = open("glove.6B.300d.txt", encoding='utf_8')
en_embeddings_index = {}  # DICT {word : vector}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    en_embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(en_embeddings_index))
# create a weight matrix for words in training docs
en_embedding_matrix = np.zeros((En_vocab_size, en_EMBEDDING_DIM))
for word, i in input_token_index.items():
    embedding_vector = en_embeddings_index.get(word)
    if embedding_vector is not None:
        en_embedding_matrix[i] = embedding_vector

# _________ Arabic Embedding __________
f = open("wiki.ar.vec", encoding='utf_8')
ar_embeddings_index = {}  # DICT {word : vector}
for line in f:
    values = line.rstrip().split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    ar_embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(ar_embeddings_index))
# create a weight matrix for words in training docs
ar_embedding_matrix = np.zeros((Ar_vocab_size, ar_EMBEDDING_DIM))
for word, i in target_token_index.items():
    embedding_vector = ar_embeddings_index.get(word)
    if embedding_vector is not None:
        ar_embedding_matrix[i] = embedding_vector

# _____ Define model input/target ____________________
encoder_input_data = np.zeros(
    (len(lines.eng), max_en),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(lines.ar), max_ar),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(lines.ar), max_ar, ar_EMBEDDING_DIM), # in one-hot .. (len(lines.ar), max_ar, num_decoder_tokens)
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(lines.eng, lines.ar)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, :] = ar_embedding_matrix[target_token_index[word] , :]
			# in one-hot .. 
			#decoder_target_data[i, t - 1, target_token_index[word]] = 1.

# ______________ Model __________________________________________

lstm_units = 1024
epoch = 60
batch_size = 128
# ____________________________Encoder __________________________
encoder_inputs = Input(shape=(None,), dtype='float32')
en_x = Embedding(En_vocab_size,
                 en_EMBEDDING_DIM,
                 weights=[en_embedding_matrix],
                 input_length=MAX_input_LENGTH,
                 trainable=False)
en = en_x(encoder_inputs)

encoder = LSTM(lstm_units, return_state=True, dtype='float32')

encoder_outputs, state_h, state_c = encoder(en)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# __________________________Decoder_____________________________
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,), dtype='float32')

dex = Embedding(Ar_vocab_size,
                ar_EMBEDDING_DIM,
                weights=[ar_embedding_matrix],
                input_length=MAX_output_LENGTH,
                trainable=False)

final_dex = dex(decoder_inputs)

decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, dtype='float32')

decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)

decoder_dense = Dense(ar_EMBEDDING_DIM, activation='softmax', dtype='float32')

decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print(model.summary())

# ____________________ trainning_________________
model.compile(optimizer='rmsprop', loss='mse', metrics=['acc']) # in one-hot i used categorical_crossentropy loss function

model.fit([encoder_input_data, decoder_input_data], decoder_target_data ,
            batch_size=batch_size,
            epochs=epoch,
            validation_split=0.2)



# ______________________ inference_________________________
encoder_model = Model(encoder_inputs, encoder_states)
print(encoder_model.summary())

# __________________________________
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dex1 = Embedding(Ar_vocab_size,
                ar_EMBEDDING_DIM,
                weights=[ar_embedding_matrix],
                input_length=1,
                trainable=False)

final_dex2 = dex1(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# __________________ pridiction function____________________
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
                len(decoded_sentence) > 50):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_char
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


def str2nums(input):
    in_test = np.zeros((1, max_en))
    remove_digits = str.maketrans('', '', digits)
    exclude = set(string.punctuation)
    i = 0

    for word in input.split():
        word = word.lower()
        word = re.sub(",", ' COMMA', word)
        word = re.sub("'", '', word)

        word = ''.join(ch for ch in word if word not in exclude)

        word = word.translate(remove_digits)
        if input_token_index.get(word) == None:
            in_test[0, i] = 0  # unk
        else:
            in_test[0, i] = input_token_index.get(word)
        i = i + 1
    return in_test


actual, predicted = list(), list()
# _______________train __ with Bleu _________________
print('_______________train __ with Bleu _________________')
for seq_index in range(1000, 1100):
    input_seq = lines.at[seq_index, 'eng']
    input_seq = str2nums(input_seq)
    target_Seq = lines.at[seq_index, 'ar']
    target_Seq = target_Seq.replace('START_', '')
    target_Seq = target_Seq.replace('_END', '')
    actual.append([target_Seq.split()])
    decoded_sentence = decode_sequence(input_seq)
    predicted.append(decoded_sentence.split())
    print('-')
    print('Input sentence:', lines.at[seq_index, 'eng'])
    print('actual sentence:', target_Seq)
    print('Decoded sentence:', decoded_sentence)
    print('_________________________________')

print("Bleu Score_train:________ ")
print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

