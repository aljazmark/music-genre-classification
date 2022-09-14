import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


DATA_PATH = "data.json"
MAPPING = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def load_data(data_path):

    with open(data_path, "r") as fp:
        datas = json.load(fp)
    data = np.array(datas["mfcc"])
    labels = np.array(datas["label"])

    return  data,labels

def split_data(data,labels,train_test, train_val):
    data_train, data_test, label_train, label_test = train_test_split(data,labels, test_size=train_test)
    data_train, data_val, label_train, label_val = train_test_split(data_train,label_train, test_size=train_val)
    return data_train,data_test,data_val,label_train,label_test,label_val

def split_data4D(data,labels,train_test, train_val):
    data_train, data_test, label_train, label_test = train_test_split(data,labels, test_size=train_test)
    data_train, data_val, label_train, label_val = train_test_split(data_train,label_train, test_size=train_val)
    data_train = data_train[...,np.newaxis]
    data_test = data_test[...,np.newaxis] 
    data_val = data_val[...,np.newaxis]
    return data_train,data_test,data_val,label_train,label_test,label_val



def sequential_model(shape):
    model = keras.Sequential()

    model.add(keras.layers.Flatten(input_shape=shape))

    model.add(keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model


def cnn_model(shape):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=shape))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(128,(3,3),activation='relu',input_shape=shape))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(256,(2,2),activation='relu',input_shape=shape))
    model.add(keras.layers.MaxPool2D((2,2),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation='relu', kernel_regularizer=keras.regularizers.l2(0.1)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(10,activation='softmax'))

    return model

def rnn_model(shape):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(256,input_shape=shape,return_sequences=True))
    
    model.add(keras.layers.LSTM(128,return_sequences=True))

    model.add(keras.layers.LSTM(64))

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(10,activation="softmax"))

    return model

def model_predict(data_prediction,label_prediction,model):
    
    data_prediction = data_prediction[np.newaxis, ...]

    predictions = model.predict(data_prediction)

    prediction_max = np.argmax(predictions,axis=1)

    print("Expected: {}, Predicted: {}".format(label_prediction,prediction_max))


def model_compile(model):
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data,labels = load_data(DATA_PATH)

    data_train, data_test,data_val, label_train, label_test,label_val = split_data(data,labels,0.25, 0.2)
    data_train_4d, data_test_4d,data_val_4d, label_train_4d, label_test_4d,label_val_4d = split_data4D(data,labels,0.25, 0.2)
    
    shape=(data_train.shape[1],data_train.shape[2])
    shape4D=(data_train_4d.shape[1],data_train_4d.shape[2],data_train_4d.shape[3])

    modelSeq = sequential_model(shape)
    modelSeq = model_compile(modelSeq)
    historySeq = modelSeq.fit(data_train, label_train, validation_data=(data_val, label_val), batch_size=32, epochs=100)
    test_err_seq,test_acc_seq = modelSeq.evaluate(data_test,label_test,verbose=1)
    print("Accuracy of Sequential model: {}".format(test_acc_seq))

    #modelCnn = cnn_model(shape4D)
    #modelCnn = model_compile(modelCnn)
    #historyCnn = modelCnn.fit(data_train_4d, label_train_4d, validation_data=(data_val_4d, label_val_4d), batch_size=32, epochs=100)
    #test_err_cnn,test_acc_cnn = modelCnn.evaluate(data_test_4d,label_test_4d,verbose=1)
    #print("Accuracy of  CNN model: {}".format(test_acc_cnn))
    #odelCnn.summary()

    #modelRnn = rnn_model(shape)
    #modelRnn = model_compile(modelRnn)
    #historyRnn = modelRnn.fit(data_train, label_train, validation_data=(data_val, label_val), batch_size=32, epochs=100)
    #test_err_rnn,test_acc_rnn = modelRnn.evaluate(data_test,label_test,verbose=1)
    #print("Accuracy of  RNN model: {}".format(test_acc_rnn))
    #modelRnn.summary()
    
    data_prediction = data_test[200]
    label_prediction = label_test[200]

    print("Sequential model prediction:")
    model_predict(data_prediction,label_prediction,modelSeq)

    #print("Cnn model prediction:")
    #model_predict(data_prediction,label_prediction,modelCnn)

    #print("Rnn model prediction:")
    #model_predict(data_prediction,label_prediction,modelRnn)

    #modelSeq.save("models/modelSeq")
    #modelCnn.save("models/modelCnn")
    #modelRnn.save("models/modelRnn")
