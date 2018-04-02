__author__ = "Akshay Raman"

import datetime
import os
import random as rn
import json
import sys

import tensorflow as tf
import numpy as np
import pandas as pd

from keras import backend as K
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Conv1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, TerminateOnNaN,\
        ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.utils import plot_model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import svm

import matplotlib.pyplot as plt

class CS205FinalProject:
    ''' '''

    def __init__(self):
        ''' '''
        #Reproducible behavior in python 3.2.3+
        os.environ['PYTHONHASHSEED'] = '0'

        #suppress AVX/FMA warning
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.seed = 0
        #fixed numpy random numbers
        np.random.seed(self.seed)

        #starting core Python generated random numbers
        rn.seed(self.seed)

        self.process_count = 0
        self.session_conf = tf.ConfigProto(intra_op_parallelism_threads=self.process_count,\
                inter_op_parallelism_threads=self.process_count)

        #Well-defined tensorflow backend
        tf.set_random_seed(self.seed)

        self.session = tf.Session(graph=tf.get_default_graph(), config=self.session_conf)
        K.set_session(self.session)

        #class variables
        self.data, self.labels = None, None

        #constants
        self.label_text = ["walking", "standing", "sitting", "laying_down"]
        self.label_count = len(self.label_text)
        self.test_size = 0.2

        #0 means use all data. This can be passed as a command line argument
        self.window_size = 0

        self.TIMESTAMP = self.get_timestamp()

        self.output_dir = ".output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_timestamp(self):
        ''' return formatted timestamp '''
        return datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%dT%H%M%S')

    def log(self, x):
        ''' fancy print statement '''
        d = self.get_timestamp()
        print("%s: %s" %(d, x))

    def read_file(self, fileName):
        ''' Read the input file '''

        self.log("Reading %s" % fileName)
        dataset = pd.read_csv(fileName)
        X = dataset.iloc[:, 2:].values
        y = dataset.iloc[:, 1].values

        return X, y

    def split_train_test(self, _X, _y):
        ''' Create test sets from all subjects '''
        X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=self.test_size)
        return X_train, X_test, y_train, y_test

    def normalize_data(self, _data):
        ''' Zero mean unit varience for the data '''
        sc = StandardScaler()
        data = sc.fit_transform(_data)
        return data

    def normalize_labels(self, _labels):
        ''' Min-subtraction for the labels '''
        min_val = min(_labels)
        labels = self.session.run(tf.subtract(_labels, min_val))
        #onehotencoding
        labels = to_categorical(labels, num_classes=self.label_count)
        return labels

    def preprocess_data(self, data, labels):
        ''' split, normalize '''
        X_train, X_test, y_train, y_test = self.split_train_test(data, labels)

        y_train = self.normalize_labels(y_train)
        y_test = self.normalize_labels(y_test)

        X_train = self.normalize_data(X_train)
        X_test = self.normalize_data(X_test)

        #Uncomment this to run RNN models
        #X_test = np.atleast_3d(X_test)

        return X_train, X_test, y_train, y_test

    def run_model(self, X_train, y_train):
        ''' Create the model, train, etc'''

        #Uncomment this to run RNN models
        #X_train = np.atleast_3d(X_train)
        #dim1,dim2 = X_train.shape[1],X_train.shape[2]

        #Uncomment this to run NN models
        dim1 = X_train.shape[1]

        self.log("Creating the model...")
        model = Sequential()

        """ <MODELCODE> """
        epoch_count = 3
        batch_size = 512

        #model.add(SimpleRNN(4, activation="relu", input_shape=(dim1,dim2)))
        #model.add(LSTM(4, activation="relu", input_shape=(dim1,dim2)))
        #model.add(GRU(4, activation="relu", input_shape=(dim1,dim2)))

        model.add(Dense(units=4, kernel_initializer="uniform", activation="relu", input_shape=(dim1,)))
        model.add(Dense(units=4, kernel_initializer="uniform", activation="relu"))
        model.add(Dense(units=4, kernel_initializer="uniform", activation="softmax"))

        model.add(Dense(self.label_count, activation='softmax'))
        opt = Adam()
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        """ </MODELCODE> """

        model.summary()

        #Set callbacks...
        _patience = min(30, max(epoch_count//5, 20))
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                       patience=_patience, verbose=1, mode='auto')
        tn = TerminateOnNaN()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-7, verbose=1)
        csv_logger = CSVLogger(os.path.join(self.output_dir, '%s_training.log' % self.TIMESTAMP))
        checkpoint_path = os.path.join(self.output_dir, "weights.best.hdf5")
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callback_fns = [early_stopping, tn, csv_logger, checkpoint, reduce_lr]

        self.log("Training...")
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_count, shuffle=True,\
                validation_split=0.1, verbose=1, callbacks=callback_fns)

        return model, history

    def plot(self, model, history, test_loss, test_acc):
        ''' Plot the graph '''

        output_dir = self.output_dir

        plot_model(model, to_file=os.path.join(output_dir, "%s_model.png" % self.TIMESTAMP), show_shapes=True)
        with open(os.path.join(output_dir, "%s_model.json" % self.TIMESTAMP), 'w') as fp:
            json.dump(model.to_json(), fp)

        history_dict = history.history
        d = self.TIMESTAMP

        def _graph(plt, d, _x, _y, _z, quantity):
            self.log("Plotting the %s graph..." % quantity)
            epochs = range(1, len(_x) + 1)
            plt.figure()
            plt.plot(epochs, _x, 'b', label='Training %s' % quantity)
            plt.plot(epochs, _y, 'r', label='Validation %s' % quantity)
            plt.plot(epochs, _z, 'g', label='Testing %s' % quantity)

            plt.title('Training, validation and testing %s' % quantity)
            plt.xlabel('epoch')
            plt.ylabel(quantity)
            plt.legend()
            #plt.show()

            file_name = os.path.join(output_dir, "%s_%s_graph.png" %(d, quantity))
            plt.savefig(file_name, bbox_inches='tight')
            self.log("Graph saved as %s" % file_name)

        #Plot loss
        train_loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        test_loss_values = [test_loss] * len(train_loss_values)
        _graph(plt, d, train_loss_values, val_loss_values, test_loss_values, "loss")

        #Plot accuracy
        train_acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
        test_acc_values = [test_acc] * len(train_acc_values)
        _graph(plt, d, train_acc_values, val_acc_values, test_acc_values, "accuracy")

    def test_model(self, model, X_test, y_test):
        ''' Evaluate the model '''
        self.log("Evaluating...")
        loss, acc = model.evaluate(X_test, y_test)
        self.log("Results: ")
        return loss, acc

    def analyze(self, model, X_test, y_test):
        ''' Display confusion matrix, perform 5-fold cross validation '''
        y_pred = model.predict(X_test)
        y_true = [np.argmax(i) for i in y_test]
        y_pred = [np.argmax(i) for i in y_pred]

        conf_matrix = confusion_matrix(y_true, y_pred)
        self.log("Confusion matrix: ")
        print(conf_matrix)


    def cleanup(self):
        ''' Cleanup '''
        K.clear_session()

    def run_main(self):
        ''' RUN! '''

        start = datetime.datetime.now()
        self.log("Starting...")
        arg_len = len(sys.argv)

        if arg_len < 2 or arg_len > 4:
            print("Format: %s <train-data-file> <test-data-file>? <window-size>?" % sys.argv[0])
            sys.exit(1)

        training_data_file = sys.argv[1]
        testing_data_file = sys.argv[2] if arg_len >= 3 else ""

        s_data, s_labels = None, None
        X_train, X_test, y_train, y_test = None, None, None, None

        #Use
        if arg_len == 2:
            self.log("Using training data file: %s" % training_data_file)
            s_data, s_labels = self.read_file(training_data_file)
            X_train, X_test, y_train, y_test = self.preprocess_data(s_data, s_labels)

        elif arg_len == 3 and sys.argv[2].isdigit():
            self.window_size = int(sys.argv[2])
            self.log("Using window size: %s" % self.window_size)
            self.log("Using training data file: %s" % training_data_file)
            s_data, s_labels = self.read_file(training_data_file)
            X_train, X_test, y_train, y_test = self.preprocess_data(s_data, s_labels)

        elif arg_len >= 3:
            if arg_len==4 and sys.argv[3].isdigit():
                self.window_size = int(sys.argv[3])
                self.log("Using window size: %s" % self.window_size)
            self.log("Using training data file: %s" % training_data_file)
            self.log("Using testing data file: %s" % testing_data_file)

            X_train, y_train = self.read_file(training_data_file)
            X_train = self.normalize_data(X_train)
            y_train = self.normalize_labels(y_train)

            X_test, y_test = self.read_file(testing_data_file)
            X_test = self.normalize_data(X_test)
            y_test = self.normalize_labels(y_test)

        if self.window_size != 0:
            X_test = X_test[:self.window_size]
            y_test = y_test[:self.window_size]

        model, history = self.run_model(X_train, y_train)

        h = history.history
        val_loss, val_acc = h['val_loss'][-1], h['val_acc'][-1]
        train_loss, train_acc = h['loss'][-1], h['acc'][-1]
        test_loss, test_acc = self.test_model(model, X_test, y_test)

        self.log("Training Loss: %s, Training Accuracy: %s" %(train_loss, train_acc))
        self.log("Validation Loss: %s, Validation Accuracy: %s" %(val_loss, val_acc))
        self.log("Testing Loss: %s, Testing Accuracy: %s" %(test_loss, test_acc))

        self.analyze(model, X_test, y_test)

        self.plot(model, history, test_loss, test_acc)

        #Non-essentials follow
        print("")
        end = datetime.datetime.now()
        diff = end-start
        self.log("Time taken: %s" %(diff))

        self.log("Done...")

        self.cleanup()

if __name__ == "__main__":
    ''' '''
    x = CS205FinalProject()
    x.run_main()
