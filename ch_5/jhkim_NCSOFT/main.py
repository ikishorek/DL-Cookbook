import random
import numpy as np
from keras.utils import get_file
from ops import *
from utils import *
from time import time

tf.enable_eager_execution()

def char_rnn_model(num_chars, num_layers, num_nodes=512, rate=0.1) :

    model = tf.keras.Sequential()

    for i in range(num_layers) :
        model.add(LSTM(num_nodes))
        model.add(dropout(rate))

    model.add(dense(num_chars))
    model.add(TimeDistributed()(softmax()))


    return model

def data_generator(all_text, char_to_idx, batch_size, chunk_size):
    X = np.zeros((batch_size, chunk_size, len(char_to_idx)))
    y = np.zeros((batch_size, chunk_size, len(char_to_idx)))

    for row in range(batch_size):
        idx = random.randrange(len(all_text) - chunk_size - 1)
        chunk = np.zeros((chunk_size + 1, len(char_to_idx)))
        for i in range(chunk_size + 1):
            chunk[i, char_to_idx[all_text[idx + i]]] = 1
        X[row, :, :] = chunk[:chunk_size]
        y[row, :, :] = chunk[1:]
    # yield X, y

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    return X, y

def generate_output(model, training_text, start_index=None, diversity=1.0, amount=400):
    if start_index is None:
        start_index = random.randint(0, len(training_text) - CHUNK_SIZE - 1)
    fragment = training_text[start_index: start_index + CHUNK_SIZE]
    generated = fragment
    for i in range(amount):
        x = np.zeros((1, len(generated), len(chars)))
        for t, char in enumerate(generated):
            x[0, t, char_to_idx[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        if diversity is None:
            next_index = int(np.argmax(preds[len(generated) - 1]))
        else:
            preds = np.asarray(preds[len(generated) - 1])
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            next_index = int(np.argmax(probas))

        next_char = chars[next_index]

        generated += next_char
        # fragment = fragment[1:] + next_char

    return generated

path = get_file('shakespeare', 'https://storage.googleapis.com/deep-learning-cookbook/100-0.txt')
shakespeare = open(path, encoding='UTF8').read()
training_text = shakespeare.split('\nTHE END', 1)[-1]

chars = list(sorted(set(training_text)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}

#################################################################

""" parameters """
BATCH_SIZE = 256
CHUNK_SIZE = 160
learning_rate = 0.01

""" dataset """
x, y = data_generator(training_text, char_to_idx, batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE)
training_epochs = 1
training_iterations = int(2 * len(training_text) / (BATCH_SIZE * CHUNK_SIZE))

""" Graph Input using Dataset API """
dataset = tf.data.Dataset.from_tensor_slices((x, y)). \
    shuffle(buffer_size=100000). \
    prefetch(buffer_size=BATCH_SIZE). \
    batch(BATCH_SIZE). \
    repeat()

iterator = dataset.make_one_shot_iterator()

""" Training """
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

""" Model """
network = char_rnn_model(len(chars), num_layers=2, num_nodes=640, rate=0)

""" Writer """
checkpoint_dir = 'checkpoints'
logs_dir = 'logs'

model_dir = 'rnn'

checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
check_folder(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, model_dir)
logs_dir = os.path.join(logs_dir, model_dir)


checkpoint = tf.train.Checkpoint(dnn=network)

# create writer for tensorboard
summary_writer = tf.contrib.summary.create_file_writer(logdir=logs_dir)
start_time = time()

# restore check-point if it exits
could_load, checkpoint_counter = load(network, checkpoint_dir)
global_step = tf.train.create_global_step()

if could_load:
    start_epoch = (int)(checkpoint_counter / training_iterations)
    start_iteration = checkpoint_counter - start_epoch * training_iterations
    counter = checkpoint_counter
    global_step.assign(checkpoint_counter)
    print(" [*] Load SUCCESS")
else:
    start_epoch = 0
    start_iteration = 0
    counter = 0
    print(" [!] Load failed...")

with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():  # for tensorboard
    for epoch in range(start_epoch, training_epochs):
        for idx in range(start_iteration, training_iterations):
            train_input, train_label = iterator.get_next()
            grads = grad(network, train_input, train_label)
            optimizer.apply_gradients(grads_and_vars=zip(grads, network.variables), global_step=global_step)

            train_loss = loss_fn(network, train_input, train_label)
            train_accuracy = accuracy_fn(network, train_input, train_label)

            tf.contrib.summary.scalar(name='train_loss', tensor=train_loss)
            tf.contrib.summary.scalar(name='train_accuracy', tensor=train_accuracy)

            print(
                "Epoch: [%2d] [%5d/%5d] time: %4.4f, train_loss: %.8f, train_accuracy: %.4f" \
                % (epoch, idx, training_iterations, time() - start_time, train_loss, train_accuracy))

            counter += 1

    checkpoint.save(file_prefix=checkpoint_prefix + '-{}'.format(counter))

    # for line in generate_output(network, training_text) :
    #     print(line)

""" Test phase """
"""
_, _ = load(network, checkpoint_dir)
for line in generate_output(network, training_text):
    print(line)
"""