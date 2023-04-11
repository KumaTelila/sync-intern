import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Load data from intents.json
with open("intents.json") as file:
    data = json.load(file)

# Load or preprocess data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [LancasterStemmer().stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [LancasterStemmer().stem(w.lower()) for w in doc]

        for w in words:
            bag.append(1) if w in wrds else bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

# Create neural network model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Load or train the model
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Define function for bag of words representation
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [LancasterStemmer().stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Chat with the bot
def chat(input_text):
    results = model.predict([bag_of_words(input_text, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    bot_response = random.choice(responses)
    return bot_response

# Handler function to handle incoming messages from Telegram
def handle_message(update, context):
    user_input = update.message.text
    # Call your modified chat function to get bot response
    bot_response = chat(user_input)
    # Send the bot response back to Telegram
    update.message.reply_text(bot_response)

# Set up Telegram bot handlers
updater = Updater(token="6088179273:AAGEuPtYlMxHzsPHpRQf4Vt6BAXp5wFiT98")  # Replace with your actual bot token

# Add message handler
updater.dispatcher.add_handler(MessageHandler(Filters.text, handle_message))

# Start the bot
updater.start_polling()
updater.idle()
