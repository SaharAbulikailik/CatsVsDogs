import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam

#################################################################################################
#####################Preparataion of data########################################################
#################################################################################################
pickle_in = open("x.pkl","rb")
x = pickle.load(pickle_in)

pickle_in = open("y.pkl","rb")
y = pickle.load(pickle_in)

x=np.array(x)
x=x.reshape(x.shape[0],-1)
x=x.T
x=x/255
y=np.array(y)
y=y.reshape((1,y.shape[0]))

x=x.T
y=y.T

##############################################################################################
######################## TRAINING ############################################################
##############################################################################################
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation


model = Sequential()


model.add(Dense(750,activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.00003)))
model.add(Dropout(0.4))

model.add(Dense(350,activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.00003)))
model.add(Dropout(0.4))

model.add(Dense(150,activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.00003)))
model.add(Dropout(0.4))


model.add(Dense(50,activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.00003)))
model.add(Dropout(0.4))


model.add(Dense(1, activation='sigmoid'))

opt = Adam(lr=0.00008)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x, y, batch_size=30, epochs=20, validation_split=0.3, shuffle=True)

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


################################################################################################
########################## testing #######################################################
################################################################################################
###################################################################################################3
####################################################################################################

x_test=pickle.load(open('x_test.pkl','rb'))
y_test=pickle.load(open('y_test.pkl','rb'))

x_test=np.array(x_test)
x_test=x_test.reshape(x_test.shape[0],-1)
x_test=x_test.T
x_test=x_test/255
y_test=np.array(y_test)
y_test=y_test.reshape((1,y_test.shape[0]))

x_test=x_test.T
y_test=y_test.T

p_test=model.predict(x_test)

########################################################################
###################confusion matrix####################################
######################################################################
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_pred_thresholded = np.where(p_test > 0.5, 1, 0)
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_thresholded)

# Visualize the confusion matrix using a heatmap
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()