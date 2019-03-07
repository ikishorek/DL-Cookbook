import tensorflow as tf

def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return loss

def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)

def accuracy_fn(model, images, labels):
    logits = model(images, training=False)
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy

def LSTM(num_nodes) :
    return tf.keras.layers.LSTM(units=num_nodes, return_sequences=True)

def dropout(rate) :
    return tf.keras.layers.Dropout(rate)

def dense(label_dim) :
    return tf.keras.layers.Dense(units=label_dim)

def softmax() :
    return tf.keras.layers.Activation(tf.keras.activations.softmax)

def TimeDistributed() :
    return tf.keras.layers.TimeDistributed
