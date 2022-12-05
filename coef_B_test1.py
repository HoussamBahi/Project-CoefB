## LIBRARIES IMPORT
print("\n----------------------------------------")
print("Importing libraries...")

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import numpy as np
import random
import tensorflow as tf
import sklearn.preprocessing as prepro
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import collections as coll
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping


## SEED FIX
print("\n----------------------------------------")
print("Fixing seed...")
random_state = 42
seed = np.random.seed(random_state)
tf.random.set_seed(random_state)
print("  -> Some random number to check if seed fix works: %f (numpy) ; %f (tf)"%(np.random.random(), tf.random.uniform((1,1))[0][0]))

## SAVE PATH
NAME = "test11_ann_50_50_b1_mse_after_test10" # Name of the current test
SAVE_PATH = "C:\\_DATA\\code_virial_ann_arXiv\\results\\" + NAME +"\\" # path where results are saved
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH + "\\model_intermediate\\", exist_ok=True)
os.makedirs(SAVE_PATH + "\\model_final\\", exist_ok=True)


## LOADING DATA
print("\n----------------------------------------")
print("Loading data...")

# Paths and format
PATH = "C:\\_DATA\\code_virial_ann_arXiv\\"
VirialFullPath = PATH + "\\dataset_virial\\allVirial-Mix.csv"
CriticalPath = PATH + "\\dataset_phy\\CriticalPropreties_v2.csv"
DipolePath = PATH + "\\dataset_phy\\DipoleMomentYaws_v0.csv"
delimiter=";"

# Loading data
CriticalCsv = np.loadtxt(CriticalPath, dtype='str',delimiter=delimiter)
CriticalLeg = CriticalCsv[0,:] #extracting legends
CriticalNom = CriticalCsv[1:,0:3] #extracting nomenclature CriticalNom = [Name,Formula,#CAS]
CriticalData = CriticalCsv[1:,3:9].astype(float) #extracting data CriticalData = [Molwt, TcK, PcMPa, Vcm3/kmol, Zc, AcentricFactor]
CriticalRef = CriticalCsv[1:,9] #extracting references

DipoleCsv = np.loadtxt(DipolePath, dtype='str',delimiter=delimiter)
DipoleLeg = DipoleCsv[0,:] #extracting legends
DipoleNom = DipoleCsv[1:,0:4] #extracting nomenclature & state DipoleNom = [Name,Formula,#CAS,state]
DipoleData = DipoleCsv[1:,4].astype(float) #extracting data DipoleData = [DipoleMomentDebye]
DipoleRef = DipoleCsv[1:,5] #extracting references

VirialPath = VirialFullPath
VirialCsv = np.loadtxt(VirialPath, dtype='str',delimiter=delimiter)
VirialLeg = VirialCsv[0,:] #extracting legends
VirialNom = VirialCsv[1:,0:4] #VirialNom = [Formula1,CASno1,Formula2,CASno2]
VirialRef = VirialCsv[1:,7] #VirialRef = [ref]

VirialUncertainties = VirialCsv[1:,6].astype(float) #VirialUncertainties = [Uncertainties]
VirialData = VirialCsv[1:,4:6].astype(float) #VirialData = [T (K),B12 (cm3/mol)]

# todo: 1 use only Tr, Tc, Pc, and ω (Dinicola)
## NEURAL NETWORK INPUT/OUTPUT
print("\n----------------------------------------")
print("Creating neural network input/output...")

def get_CriticalPropreties (CAS, CriticalNom, CriticalData):
    try:
        ind=np.where(CriticalNom[:,2]==CAS)[0][0]
        return CriticalData[ind]
    except:
        return np.zeros((6), dtype=bool)

def get_DipoleMoment (CAS, DipoleNom, DipoleData):
    try:
        ind=np.where(DipoleNom[:,2]==CAS)[0][0]
        return DipoleData[ind]
    except:
        return ('False')

X = []
Y = []
molecules_with_unfound_Critical_propreties=[]
molecules_with_unfound_Dipole_Moment=[]
mix_with_big_uncert=[]
nb_of_unusable_data = 0

for i in range (len(VirialData)):
    CAS1 = VirialNom[i,1]
    CAS2 = VirialNom[i,3]
    if random.random()>0.5:
        CAS1,CAS2 = CAS2,CAS1 #random shuffle
    Critical1 = get_CriticalPropreties (CAS1, CriticalNom, CriticalData)
    Critical2 = get_CriticalPropreties (CAS2, CriticalNom, CriticalData)
    Dipole1 = get_DipoleMoment (CAS1, DipoleNom, DipoleData)
    Dipole2 = get_DipoleMoment (CAS2, DipoleNom, DipoleData)
    if (Critical1[0]!=False and Critical2[0]!=False and Dipole1!='False' and Dipole2!='False' and abs(VirialUncertainties[i])<50):
        #X.append(np.concatenate((Critical1,[Dipole1],Critical2,[Dipole2],[VirialData[i,0]],[VirialUncertainties[i]])))
        #X.append(np.concatenate((Critical1,[Dipole1],Critical2,[Dipole2],[VirialData[i,0]])))
        #X.append(np.concatenate((Critical1[1:],[Dipole1],Critical2[1:],[Dipole2],[VirialData[i,0]])))
        #X.append(np.concatenate((Critical1[1:4],[Critical1[5]],[Dipole1],Critical2[1:4],[Critical2[5]],[Dipole2],[VirialData[i,0]])))
        X.append(np.concatenate((Critical1[1:4],[Critical1[5]],[Dipole1],Critical2[1:4],[Critical2[5]],[Dipole2],[VirialData[i,0]/Critical1[1]])))
        Y.append(VirialData[i,1])
    else:
        nb_of_unusable_data+=1
        if (Critical1[0]==False):
            molecules_with_unfound_Critical_propreties.append(CAS1)
        if (Critical2[0]==False):
            molecules_with_unfound_Critical_propreties.append(CAS2)
        if (Dipole1=='False'):
            molecules_with_unfound_Dipole_Moment.append(CAS1)
        if (Dipole2=='False'):
            molecules_with_unfound_Dipole_Moment.append(CAS2)
        if (abs(VirialUncertainties[i])>50):
             mix_with_big_uncert.append((CAS1,CAS2))

X = np.array(X)
Y = np.array(Y)
N, inpSize = X.shape
print("Unusable data: %.2f %s" %(100*nb_of_unusable_data/len(VirialData),'%'))



## NORMALIZING DATA
print("\n----------------------------------------")
print("Normalizing data...")
# todo: 2 use normalization

# scalerX = prepro.MinMaxScaler()
# scalerX.fit(X)
# normalizedX = scalerX.transform(X)
# X = normalizedX
#
# Y = Y.reshape(-1, 1)
# scalerY = prepro.MinMaxScaler()
# scalerY.fit(Y)
# normalizedY = scalerY.transform(Y)
# Y = normalizedY

# inverse transform
# inverse = scaler.inverse_transform(normalizedX)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True)


## NEURAL NETWORK ARCHITECTURE
print("\n----------------------------------------")
print("Neural network architecture...")

#todo: 4 struct optimization (2 x 19 in Dinicola)
def architecture(inpSize,nb_laters = 2, nb_neurons=50):
    initializer="normal"
    model = tf.keras.Sequential()
    for i in range (nb_laters):
        model.add(tf.keras.layers.Dense(nb_neurons, input_dim=inpSize, kernel_initializer=initializer, activation='relu'))
    model.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
    return model


## HYPERPARAMETERS
print("\n----------------------------------------")
print("Hyperparameters...")

# BATCH SIZE
BATCH_SIZE = 1
# todo: optimize batch size ?

# OPTIMIZER
OPTIMIZER = tf.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) #todo: optimize optimizer
# OPTIMIZER = tf.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
# OPTIMIZER = tf.optimizers.Adam(lr=1e-3)
# OPTIMIZER = tf.optimizers.SGD(lr=0.01, clipnorm=1.)
#OPTIMIZER = tf.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


# LOSS
LOSS = tf.keras.losses.MSE
#LOSS = "mean_absolute_percentage_error"
#todo: 3 optimize loss

# EPOCHS
EPOCHS = 10000

# MODEL
model = architecture(inpSize)
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[tf.keras.losses.MSE, "mean_absolute_percentage_error"])
model.summary()
# pretrained weights:
model.load_weights("C:\\_DATA\\code_virial_ann_arXiv\\results\\test10_ann_50_50_b1_mpe\\model_intermediate\\test10_ann_50_50_b1_mpe_1501.h5")
# todo: try train using MSE after MPE

## CALLBACKS
model_checkpoint = ModelCheckpoint(SAVE_PATH + "\\model_intermediate\\"+ NAME + '_{epoch:02d}'+'.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=False,
                                    save_weights_only=True)
earlystopper = EarlyStopping(monitor='val_loss',mode='min',  patience=4000, restore_best_weights=True, verbose=1)

tensorboard = TensorBoard(SAVE_PATH+'\\logs\\{}',
                               profile_batch=0)


## TRAINING
print("\n----------------------------------------")
print("Training...")

t1 = time.time()
# Fit the model on the dataset
history = model.fit(x = X_train, y = Y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_test, Y_test),
                    verbose = 2,
                    callbacks=[model_checkpoint, earlystopper, tensorboard])

t2 = time.time()
computing_time = t2-t1
print('\n')
print('Training Time =', computing_time, 's')

#python3 -m tensorboard.main --logdir=logs --port=6006

## SAVING FINAL MODEL
print("\n----------------------------------------")
print("Saving final model...")

model.save_weights(SAVE_PATH + "\\model_final\\" + NAME +'_final_weights.h5')
model.save(SAVE_PATH + "\\model_final\\" + NAME +'_my_model.h5')  # creates a HDF5 file 'my_model.h5'
model_json = model.to_json()

with open(SAVE_PATH + "\\model_final\\" + NAME +"_model.json", "w") as json_file:
    json_file.write(model_json)

print("Saved model to disk")

## MODEL PERFORMANCES
print("\n----------------------------------------")
print("Assessing final model performances...")

preds = model.predict(X_test)[:,0]

#Inverse normalization
# Y_test =  scalerY.inverse_transform(Y_test)
# preds = preds.reshape(-1, 1)
# preds = scalerY.inverse_transform(preds)

ME = np.mean(abs(preds - Y_test))
RMSE = np.sqrt(np.mean((preds - Y_test)**2))
MPE = np.mean(100*(abs((preds-Y_test)/Y_test))**1)
print("ME = %.2f cm3/mol" %(ME))
print("RMSE = %.2f cm3/mol" %(RMSE))
print("MPE = %.2f percts" %(MPE))

t2=time.time()
print("Comput. time = %.2f s"%(t2-t1))

## BEST MODEL PERFS


## TRAINING GRAPHS

epochs_loss_train = np.array(history.history['loss'])
epochs_loss_val = np.array(history.history['val_loss'])
plt.figure()
plt.title("Loss")
plt.plot(epochs_loss_train, 'b-', label = "training (best %.2f %% at epoch %i)"%(epochs_loss_train.min(),epochs_loss_train.argmin()))
plt.plot(epochs_loss_val, 'r-', label = "validation (best %.2f %% at epoch %i)"%(epochs_loss_val.min(),epochs_loss_val.argmin()))
plt.legend()
plt.savefig(SAVE_PATH+"epochs_loss.png")
plt.show()

# RMSE graph
epochs_RMSE_train = np.sqrt(history.history['mean_squared_error'])
epochs_RMSE_val = np.sqrt(history.history['val_mean_squared_error'])
plt.figure()
plt.title("RMSE")
plt.plot(epochs_RMSE_train, 'b-', label = "training (best %.2f cm3/mol at epoch %i)"%(epochs_RMSE_train.min(),epochs_RMSE_train.argmin()))
plt.plot(epochs_RMSE_val, 'r-', label = "validation (best %.2f cm3/mol at epoch %i)"%(epochs_RMSE_val.min(),epochs_RMSE_val.argmin()))
plt.legend()
plt.savefig(SAVE_PATH+"epochs_RMSE.png")
plt.show()

# MPE graph
epochs_MPE_train = np.sqrt(history.history['mean_absolute_percentage_error'])
epochs_MPE_val = np.sqrt(history.history['val_mean_absolute_percentage_error'])
plt.figure()
plt.title("MPE")
plt.plot(epochs_MPE_train, 'b-', label = "training (best %.2f %% at epoch %i)"%(epochs_MPE_train.min(),epochs_MPE_train.argmin()))
plt.plot(epochs_MPE_val, 'r-', label = "validation (best %.2f %% at epoch %i)"%(epochs_MPE_val.min(),epochs_MPE_val.argmin()))
plt.legend()
plt.savefig(SAVE_PATH+"epochs_MPE.png")
plt.show()


## Predict B12 of some molecule pairs

def pair_prediction(model,CAS1,CAS2,CriticalNom,CriticalData,DipoleNom,DipoleData, save=False):
    crit1 = get_CriticalPropreties (CAS1, CriticalNom, CriticalData)
    crit2 = get_CriticalPropreties (CAS2, CriticalNom, CriticalData)
    dip1 = get_DipoleMoment (CAS1, DipoleNom, DipoleData)
    dip2 = get_DipoleMoment (CAS1, DipoleNom, DipoleData)
    temps = np.arange(100,600,2)
    Xpp=[]
    for T in temps:
        #xpp = np.concatenate((crit1,[dip1],crit2,[dip2],[T],[1]))
        xpp = np.concatenate((crit1[1:4],[crit1[5]],[dip1],crit2[1:4],[crit2[5]],[dip2],[T/crit1[1]]))
        #xpp = np.concatenate((crit1,[dip1],crit2,[dip2],[T]))
        Xpp.append(xpp)
    Xpp = np.array(Xpp)
    # normalizedXpp = scalerX.transform(Xpp)
    # normalizedYpp = model.predict(normalizedXpp)[:,0]
    # normalizedYpp = normalizedYpp.reshape(-1, 1)
    # Ypp = scalerY.inverse_transform(normalizedYpp)
    Ypp = model.predict(Xpp)[:,0]
    fig = plt.figure(figsize=(9,5))
    plt.title("pred of %s + %s"%(CAS1,CAS2))
    #ploting exp data
    for i in range (len(VirialNom)):
        if ((CAS1==VirialNom[i,1] and CAS2==VirialNom[i,3]) or (CAS2==VirialNom[i,1] and CAS1==VirialNom[i,3])):
            plt.plot(VirialData[i,0],VirialData[i,1],'r+',label="exp: "+VirialRef[i])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    plt.plot(temps, Ypp,'k-.', label="B12 prediction for "+CAS1+" & "+CAS2)
    if save is not False:
        plt.savefig(save)
    plt.show()

best_model_loss = architecture(inpSize)
best_model_loss.load_weights(SAVE_PATH+"\\model_intermediate\\"+NAME+"_"+str(epochs_loss_val.argmin())+".h5")

#Ne:7440-01-9, Xe:7440-63-3
pair_prediction(best_model_loss,"7440-01-9","7440-63-3",CriticalNom,CriticalData,DipoleNom,DipoleData, save=SAVE_PATH+"Ne_Xe_pred_bestloss.png")
#CO2:124-38-9, O2:7782-44-7
pair_prediction(best_model_loss,"124-38-9","7782-44-7",CriticalNom,CriticalData,DipoleNom,DipoleData, save=SAVE_PATH+"CO2_O2_pred_bestloss.png")

best_model_RMSE = architecture(inpSize)
best_model_RMSE.load_weights(SAVE_PATH+"\\model_intermediate\\"+NAME+"_"+str(epochs_RMSE_val.argmin())+".h5")

#Ne:7440-01-9, Xe:7440-63-3
pair_prediction(best_model_RMSE,"7440-01-9","7440-63-3",CriticalNom,CriticalData,DipoleNom,DipoleData, save=SAVE_PATH+"Ne_Xe_pred_bestRMSE.png")
#CO2:124-38-9, O2:7782-44-7
pair_prediction(best_model_RMSE,"124-38-9","7782-44-7",CriticalNom,CriticalData,DipoleNom,DipoleData, save=SAVE_PATH+"CO2_O2_pred_bestRMSE.png")








