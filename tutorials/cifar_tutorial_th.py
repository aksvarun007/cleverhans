'''Train a simple deep CNN on the CIFAR10  dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar_tutorial_th.py

'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from cleverhans.utils_th import th_model_train, th_model_eval,batch_eval
from cleverhans.attacks_th import fgsm
from keras.models import Sequential
from keras.utils.visualize_util import plot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', default=500, help='Size of training batches')
parser.add_argument('--nb_epochs', '-e', default=70, type=int, help='Number of epochs to train model')
parser.add_argument('--learning_rate', '-lr', default=0.01, type=float, help='Learning rate for training')
args = parser.parse_args()

# ################## Download and prepare the CIFAR dataset ##################
# This is just some way of getting the CIFAR dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    #X_train_flip = X_train[:,:,:,::-1]
    #Y_train_flip = Y_train
    #X_train = np.concatenate((X_train,X_train_flip),axis=0)
    #Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=(X_train).astype(keras.backend.floatx()),
        Y_train=Y_train.astype('int32'),
        X_test = (X_test).astype(keras.backend.floatx()),
        Y_test = Y_test.astype('int32'),)



nb_classes = 10

# The data, shuffled and split between train and test sets:
data = load_data()
X_train = data['X_train']
y_train = data['Y_train']
X_test = data['X_test']
y_test = data['Y_test']


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
def cifar_net(nb_classes=10,img_rows=32,img_cols=32,img_channels=3):

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
                            
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3,border_mode='same'))
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu')) 
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model


x_shape = (None, 3, 32, 32)
y_shape = (None, 10)
x = T.tensor4('x')
y = T.matrix('y')

model = cifar_net()
model.build(x_shape)
predictions = model(x)
print("Defined Theano model graph.")

def evaluate():
    # Evaluate the accuracy of the CIFAR model on legitimate test examples
    accuracy = th_model_eval(x, y, predictions, X_test, Y_test, args=args)
    assert X_test.shape[0] == 10000, X_test.shape
    print('Test accuracy on legitimate test examples: ' + str(accuracy))
    pass

# Train a CIFAR model
th_model_train(x, y, predictions, model.trainable_weights, X_train, Y_train, evaluate=evaluate, args=args)
# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
adv_x = fgsm(x, predictions, eps=0.3)
X_test_adv, = batch_eval([x], [adv_x], [X_test], args=args)
assert X_test_adv.shape[0] == 10000, X_test_adv.shape

# Evaluate the accuracy of the CIFAR model on adversarial examples
accuracy = th_model_eval(x, y, predictions, X_test_adv, Y_test, args=args)
print('Test accuracy on adversarial examples: ' + str(accuracy))



print("Repeating the process, using adversarial training")
x_2 = T.tensor4('x_2')
y_2 = T.matrix('y_2')
model_2= cifar_net()
model_2.build(x_shape)
predictions_2 = model_2(x_2)
adv_x_2 = fgsm(x_2, predictions_2, eps=0.3)
predictions_adv_2 = model_2(adv_x_2)

def evaluate_2():
    # Evaluate the accuracy of the adversarialy trained CIFAR model on
    # legitimate test examples
    accuracy = th_model_eval(x_2, y_2, predictions_2, X_test, Y_test, args=args)
    print('validation accuracy on legitimate examples: ' + str(accuracy))
    # Evaluate the accuracy of the adversarially trained CIFAR model on
    # adversarial examples
    accuracy_adv = th_model_eval(x_2, y_2, predictions_adv_2, X_test, Y_test, args=args)
    print('Validation accuracy on adversarial examples: ' + str(accuracy_adv))

# Perform adversarial training
th_model_train(x_2, y_2, predictions_2, model_2.trainable_weights, X_train, Y_train, predictions_adv=predictions_adv_2,evaluate=evaluate_2, args=args)        



