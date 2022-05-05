# Hinge Loss
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2 
from matplotlib import pyplot
# generate 2d classification dataset
X, y = make_circles(n_samples=5000, noise=0.1, random_state=1)
# split into train and test
train1 = 2500
trainX, testX = X[:train1, :], X[train1:, :]
trainy, testy = y[:train1], y[train1:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='tanh'))
opt = gradient_descent_v2.SGD(learning_rater=0.01, momentum=0.9)
model.compile(loss='hinge', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=50, verbose=0)
# evaluate the model
train_acc = model.evaluate(trainX, trainy, verbose=0)
test_acc = model.evaluate(testX, testy, verbose=0)
# plot loss during training
pyplot.title('Hinge Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()