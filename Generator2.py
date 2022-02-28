###  IMPORT  ###
from glob import glob
from tqdm import tqdm
import cv2
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np

### PARAMETRES ###
optimizer = Adam(lr=0.0002, beta_1=0.5)
Batch_Size= 1000

### RECUP DATA ###
dossier_images = "im"

nom_images = glob(dossier_images + "/*.jpg")
for nom_de_limage_actuel_a_traiter in tqdm(nom_images):
    image = cv2.imread(nom_de_limage_actuel_a_traiter, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 100))
    image = (image.astype("float32") - 127.5) / 127.5

### CREER LE DISCRIMINATOR ###
Discriminator = Sequential()

Discriminator.add(Conv2D(16, kernel_size=(4, 4),
                         activation='relu',
                         input_shape=(100, 100, 3)))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Conv2D(32, (3, 3), activation='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Conv2D(64, (3, 3), activation='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Conv2D(128, (3, 3), activation='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Flatten())
Discriminator.add(Dense(128, activation='relu'))
Discriminator.add(Dense(2, activation='sigmoid'))

print("===== DISCRIMINATOR =====")
Discriminator.summary()
Discriminator.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

###   CREER LE GENERATOR   ###
bruit = np.random.normal(0, 1, (Batch_Size, 100))

Generator = Sequential()
Generator.add(Dense(25*25*128, activation='relu', input_shape=(100,)))

Generator.add(Reshape((25, 25, 128)))
#Entrée : 25, 25, 128
Generator.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
Generator.add(UpSampling2D(size=(2, 2))) #25x25 -> 50x50

Generator.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
Generator.add(UpSampling2D(size=(2, 2))) #50x50 -> 100x100

Generator.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))
#couche de sortie. 100, 100, 3
Generator.add(Conv2D(3, (2, 2),padding='same' ,activation='tanh'))

print()
print()
print("===== GENERATOR =====")
Generator.summary()

###   ON ENTRAINE TOUT CA  ###
    # ON ENTRAINE LE DISCRIMINATOR #

    # ON ENTRAINE LE COMBO DES DEUX #

### ON SAVE SE QUIL Y A A SAVE ###