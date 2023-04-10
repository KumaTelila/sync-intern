import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load JSON data
with open('intents.json') as file:
    data = json.load(file)

# Extract data from JSON
intents = data['intents']
tags = []
patterns = []
responses = []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses.extend(intent['responses'])

# Create training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for pattern in patterns:
    token_list = tokenizer.texts_to_sequences([pattern])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Build model
model = Sequential()
model.add(Embedding(total_words, 128, input_length=max_sequence_len-1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(total_words, activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=200, batch_size=5)

# Chatbot response logic
def chatbot_response(model, tokenizer, intents, input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    input_seq = pad_sequences([input_seq], maxlen=max_sequence_len-1, padding='pre')
    prediction = model.predict(input_seq)
    tag_index = np.argmax(prediction)
    tag = tags[tag_index]
    responses_for_tag = [response for intent in intents if intent['tag'] == tag for response in intent['responses']]
    response = random.choice(responses_for_tag)
    return response

# Chat loop
while True:
    input_text = input("You: ")
    if input_text.lower() == 'quit':
        break
    response = chatbot_response(model, tokenizer, intents, input_text)
    print("Bot: ", response)
