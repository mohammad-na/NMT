import pandas as pd
import numpy as np
import string
from string import digits
import re
import sklearn
from keras.layers import Input, LSTM, Embedding, Dense, Dropout
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from matplotlib import pyplot
from keras.preprocessing.text import Tokenizer

# ............................................................................................#
# this function remove noise from word (Strip Harakat)
def deNoise(text):
    noise = re.compile(""" ø    | # Tashdid
                                                  ó    | # Fatha
                                                  ð    | # Tanwin Fath
                                                  õ    | # Damma
                                                  ñ    | # Tanwin Damm
                                                  ö    | # Kasra
                                                  ò    | # Tanwin Kasr
                                                  ú    | # Sukun
                                                  Ü     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    return text


# ..................Vocalizing English Data...........................
lines = pd.read_table('dataset.txt', names=['eng', 'ar'])

lines = lines[0:-1]

lines = sklearn.utils.shuffle(lines)
lines = lines.reset_index(drop=True)


print(lines.shape)
# cleanup
lines.eng = lines.eng.apply(lambda x: x.lower())
lines.ar = lines.ar.apply(lambda x: x.lower())
lines.ar = lines.ar.apply(lambda x: deNoise(x))

lines.eng = lines.eng.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
lines.ar = lines.ar.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))

exclude_en = set(string.punctuation)
lines.eng = lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude_en))
exclude_ar = set(['¿','!','.',',','¡'])
lines.ar = lines.ar.apply(lambda x: ''.join(ch for ch in x if ch not in exclude_ar))

remove_digits = str.maketrans('', '', digits)
lines.eng = lines.eng.apply(lambda x: x.translate(remove_digits))
lines.ar = lines.ar.apply(lambda x: x.translate(remove_digits))
lines.ar = lines.ar.apply(lambda x: re.sub('\d+','',x))

print(lines.head())

lines.ar = lines.ar.apply(lambda x: 'sos ' + x + ' eos')

en_input_data = lines["eng"].values
ar_input_data = lines["ar"].values

#___________________ all the words ________________
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

#_______________ Tokenizer _______________
num_of_en_vocab = int(len(all_eng_words) * 0.8)
num_of_ar_vocab = int(len(all_arabic_words) * 0.8)


def max_length(lines):
	return max(len(line.split()) for line in lines)

def create_tokenizer(lines,numOfWords):
	tokenizer = Tokenizer(num_words=numOfWords + 1, oov_token='unk')
	tokenizer.fit_on_texts(lines)
	return tokenizer

eng_tokenizer = create_tokenizer(en_input_data,num_of_en_vocab)
# make the dict contains only top frequent words {
eng_tokenizer.word_index = {e:i for e,i in eng_tokenizer.word_index.items() if i <= num_of_en_vocab}
eng_tokenizer.word_index[eng_tokenizer.oov_token] = num_of_en_vocab + 1
# }
num_encoder_tokens = len(eng_tokenizer.word_index) + 1
max_en = max_length(en_input_data)
print('English Vocabulary Size: %d' % num_encoder_tokens)
print('English Max Length: %d' % (max_en))

ar_tokenizer = create_tokenizer(ar_input_data,num_of_ar_vocab)
# make the dict contains only top frequent words {
ar_tokenizer.word_index = {e:i for e,i in ar_tokenizer.word_index.items() if i <= num_of_ar_vocab}
ar_tokenizer.word_index[eng_tokenizer.oov_token] = num_of_ar_vocab + 1
# }
num_decoder_tokens = len(ar_tokenizer.word_index) + 1
max_ar = max_length(ar_input_data)
print('Arabic Vocabulary Size: %d' % num_decoder_tokens)
print('Arabic Max Length: %d' % (max_ar))


encoder_input_data = eng_tokenizer.texts_to_sequences(en_input_data)
encoder_input_data = pad_sequences(encoder_input_data, max_en)

decoder_input_data = ar_tokenizer.texts_to_sequences(ar_input_data)
decoder_input_data = pad_sequences(decoder_input_data, max_ar , padding='post')

decoder_target_data = decoder_input_data[:,1:]
decoder_target_data = np.append(decoder_target_data,np.zeros(shape=(decoder_target_data.shape[0],1)),axis=1)

#______________ dictionary ______________
input_token_index = eng_tokenizer.word_index
target_token_index = ar_tokenizer.word_index


#____________________________
en_EMBEDDING_DIM = 300
ar_EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.3

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
en_embedding_matrix = np.zeros((num_encoder_tokens, en_EMBEDDING_DIM))
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
ar_embedding_matrix = np.zeros((num_decoder_tokens, ar_EMBEDDING_DIM))
for word, i in target_token_index.items():
    embedding_vector = ar_embeddings_index.get(word)
    if embedding_vector is not None:
        ar_embedding_matrix[i] = embedding_vector

# _____________________________________
lstm_units = 1024
epoch = 35
batch_size = 128

size = {"en": num_encoder_tokens, "ar": num_decoder_tokens, "en_EMBEDDING_DIM": en_EMBEDDING_DIM, "ar_EMBEDDING_DIM": ar_EMBEDDING_DIM,
        "lstm_units": lstm_units, "batch_size": batch_size,"max_en":int(max_en), "max_ar":int(max_ar)}
with open('size.json', 'w') as fp:
    json.dump(size, fp, sort_keys=True, indent=4)
fp.close()
del fp
# ____________________________Encoder __________________________
encoder_inputs = Input(shape=(None,))
en_x = Embedding(num_encoder_tokens,
                 en_EMBEDDING_DIM,
                 weights=[en_embedding_matrix],
                 input_length=max_en,
                 trainable=False)
en = en_x(encoder_inputs)

encoder1 = LSTM(lstm_units, return_state=True, return_sequences = True)
encoder2 = LSTM(lstm_units, return_state=True, return_sequences = True)
encoder = LSTM(lstm_units, return_state=True, return_sequences = True)

encoder_outputs, state_h, state_c = encoder(encoder2(encoder1(en)))
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# __________________________Decoder_____________________________
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

dex = Embedding(num_decoder_tokens,
                ar_EMBEDDING_DIM,
                weights=[ar_embedding_matrix],
                input_length=max_ar,
                trainable=False)

final_dex = dex(decoder_inputs)


decoder_lstm1 = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_lstm2 = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_lstm2(decoder_lstm1(final_dex,
                                     initial_state=encoder_states)))

drop = Dropout(0,2)(decoder_outputs)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(drop )

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print(model.summary())

# ____________________ trainning_________________
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
          batch_size=batch_size,
          epochs=epoch,
          validation_split=0.33)

# plot train and validation loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation acc')
pyplot.ylabel('acc')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

model.save_weights('NewModel_1L.h5')
# ______________________ inference_________________________
encoder_model = Model(encoder_inputs, encoder_states)
print(encoder_model.summary())

# __________________________________
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dex1 = Embedding(num_decoder_tokens,
                ar_EMBEDDING_DIM,
                weights=[ar_embedding_matrix],
                input_length=1,
                trainable=False)

final_dex2 = dex1(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_lstm2(decoder_lstm1(final_dex2, initial_state=decoder_states_inputs)))

decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

print(decoder_model.summary())

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
    target_seq[0, 0] = target_token_index['sos']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if reverse_target_char_index.get(sampled_token_index) == None:
          sampled_char = 'unk'
        else:
          sampled_char = reverse_target_char_index.get(sampled_token_index)
        # decoded_sentence += ' ' + sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'eos' or
                len(decoded_sentence) > 52):
            stop_condition = True
        # elif sampled_char == 'COMMA' :
        #    decoded_sentence += ' ' + ','
        else:
            decoded_sentence += ' ' + sampled_char
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


def str2nums(input,max_en):
    in_test = np.zeros((1, len(input.split())))
    remove_digits = str.maketrans('', '', digits)
    exclude = set(string.punctuation)
    i = 0
    for word in input.split():
        word = word.lower()
        word = deNoise(word)
        word = re.sub(",", ' COMMA', word)
        word = re.sub("'", '', word)
        word = ''.join(ch for ch in word if word not in exclude)
        word = word.translate(remove_digits)

        if input_token_index.get(word) == None:
            in_test[0, i] = input_token_index.get('unk') #unk
        else:
            in_test[0, i] = input_token_index.get(word)
        i = i + 1
    in_test = pad_sequences(in_test, max_en)
    return in_test

actual, predicted = list(), list()
# _______________train __ with Bleu _________________
for seq_index in range(3000,3100):
    input_seq = lines.at[seq_index,'eng']
    input_seq = str2nums(input_seq,max_en)
    target_Seq = lines.at[seq_index,'ar']
    target_Seq = target_Seq.replace('sos', '')
    target_Seq = target_Seq.replace('eos', '')
    actual.append([target_Seq.split()])
    decoded_sentence = decode_sequence(input_seq)
    predicted.append(decoded_sentence.split())
    print('-')
    print('Input sentence:', lines.at[seq_index,'eng'])
    print('actual sentence:', target_Seq)
    print('Decoded sentence:', decoded_sentence)
print('test M2')
print("Bleu Score_train:________ ")
print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

actual, predicted = list(), list()
# _______________test __ with Bleu _________________
for seq_index in range(8000,8100):
    input_seq = lines.at[seq_index,'eng']
    input_seq = str2nums(input_seq,max_en)
    target_Seq = lines.at[seq_index,'ar']
    target_Seq = target_Seq.replace('sos', '')
    target_Seq = target_Seq.replace('eos', '')
    actual.append([target_Seq.split()])
    decoded_sentence = decode_sequence(input_seq)
    predicted.append(decoded_sentence.split())
    print('-')
    print('Input sentence:', lines.at[seq_index,'eng'])
    print('actual sentence:', target_Seq)
    print('Decoded sentence:', decoded_sentence)
print('validation M2')
print("Bleu Score_train:________ ")
print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
