# Coding Your Generative AI Model

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Step 1: Gathering Data
data = ["This is the first sentence.", "Here's the second sentence.", "Finally, the third one."]

# Step 2: Preprocessing Your Data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(data)

input_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Step 3: Choosing a Generative Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_length-1),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),  # Experiment with Bidirectional LSTM
    tf.keras.layers.LSTM(100),
#    tf.keras.layers.Dropout(0.2),  # Experiment with Dropout for regularization
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Training Your Generative Model
model.fit(X, y, epochs=100, verbose=1)  # Adjust epochs based on validation

# Step 5: Generating Text
seed_text = "This is"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted[0])

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)