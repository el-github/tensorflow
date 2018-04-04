import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

# set the size to 32
trainX = pad_sequences(trainX, maxlen=32, value=0)
# convert to one hot form. the size is 2
trainY = to_categorical(trainY, nb_classes=2)

# initialization
tf.reset_default_graph()

# input layer. The number of nodes is 32
net = tflearn.input_data([None, 32])
# embedding layer. 5445 = types of words + 1
net = tflearn.embedding(net, input_dim=5445, output_dim=128)
# lstm block
net = tflearn.lstm(net, 128, dropout=0.5)
# output layer
net = tflearn.fully_connected(net, 2, activation='softmax')

# optimizer
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

# execute learning
model = tflearn.DNN(net)
model.fit(trainX, trainY, n_epoch=50, batch_size=32, validation_set=0.2, shuffle=True, show_metric=True)
