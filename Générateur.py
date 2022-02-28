#IMPORT

from glob import glob
from tqdm import tqdm
from cv2 import cv2
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf


config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

#RECUP DATA

dossier_image = "images"
optimizer = Adam(lr=0.0002, beta_1=0.5)
cycle = 500
Images = []
Batch = 1000

nom_images = glob(dossier_image + "/*.jpg")
for nom_images_a_traiter in tqdm(nom_images):
    image = cv2.imread(nom_images_a_traiter, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (200, 200))
    image = (image.astype('float32') - 127.5) / 255
    Images.append(image)

Images = np.asarray(Images)
#DISCRIMINATEUR

Discriminator = Sequential()
Discriminator.add(Conv2D(32, kernel_size=(4, 4),
                         activation ='relu',
                         input_shape=(200, 200, 3)))

Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Conv2D(64, (3, 3), activation ='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Conv2D(128, (3, 3), activation ='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))


Discriminator.add(Flatten())

Discriminator.add(Dense(128, activation='relu'))
Discriminator.add(Dense(1, activation='sigmoid'))

print("Discriminateur")

Discriminator.summary()
Discriminator.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
#GENERATEUR
Generator = Sequential()

Generator.add(Dense(25*25*128, activation='relu', input_shape=(100,)))
Generator.add(Reshape((25, 25, 128)))

#25*25*128
Generator.add(Conv2D(128, (2, 2), padding="same", activation='tanh'))
Generator.add(UpSampling2D(size=(2, 2)))

#50*50*64
Generator.add(Conv2D(64, (2, 2), padding="same", activation='tanh'))
Generator.add(UpSampling2D(size=(2, 2)))

#100x100x32
Generator.add(Conv2D(32, (2,2), padding="same", activation='tanh'))
Generator.add(UpSampling2D(size=(2, 2)))

#200x200x3
Generator.add(Conv2D(3, (2,2), padding="same", activation='tanh'))

print("Generator")
Generator.summary()

Combo = Sequential()

Combo.add(Generator)
Combo.add(Discriminator)


Combo.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

#ENTRAINEMENT DISCRIMINATEUR

CycleActuel = 0

while CycleActuel < cycle:
    Batch_vraies = Images[np.random.randint(0, Images.shape[0], size=Batch)]
    Labels_vraies = np.ones(Batch)

    Labels_faux = np.zeros(Batch)
    Bruit = np.random.normal(0, 1, size=[Batch, 100])

    Images_Fausses = Generator.predict(Bruit)
    Labels_faux = np.zeros(Batch)

    Images_Entrainement = np.concatenate([Batch_vraies, Images_Fausses])
    Labels_Entrainement = np.concatenate([Labels_vraies, Labels_faux])

    Discriminator.trainable = True
    print()
    print("Cycle N°" + str(CycleActuel))
    print()
    Discriminator.fit(Images_Entrainement, Labels_Entrainement, epochs=1, batch_size=200)

#ENTRAIENEMENT DU COMBO DS DEUX

    #GENERER DU BRUIT

    Batch = np.random.normal(0, 1, size=[Batch, 100])
    Labels_Combo = np.ones(Batch)
    Discriminator.trainable = False
    Combo.fit(Bruit, Labels_Combo, epochs=1, batch_size=Batch)


#SAUVEGARDE

    if CycleActuel % 5 == 0:
        bruit = np.random.normal(0, 1, size=[1, 100])
        image = Generator.predict(bruit)
        image = (image * 127.5) + 127.5
        image = image.astype("uint8")
        image = np.reshape(image, (100, 100, 3))
        NomImage = 'ImageGenere N°' + str(CycleActuel) + '.jpg'
        cv2.imwrite("doss/" + NomImage, image)
        CycleActuel += 1

    Generator.save('Generator.h5')
    Discriminator.save('Discriminator.h5')