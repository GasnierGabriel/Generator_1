###  IMPORT  ###
from glob import glob
from tqdm import tqdm
import cv2
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf

### PARAMETRES ###
optimizer = Adam(lr=0.0002, beta_1=0.5)
Batch_Size= 1000
NombreBoucle = 500

Images = []
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

dossier_image = "cats0"


nom_images = glob(dossier_image + "/*.jpg")
for nom_images_a_traiter in tqdm(nom_images):
    image = cv2.imread(nom_images_a_traiter, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (16, 16))
    image = (image.astype('float32') - 127.5) / 255
    Images.append(image)

Images = np.array(Images)

### CREER LE DISCRIMINATOR ###
Discriminator = Sequential()

Discriminator.add(Conv2D(32, kernel_size=(4, 4),
                         activation='relu',
                         input_shape=(16, 16, 3)))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Conv2D(64, (2, 2), activation='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Conv2D(64, (3, 3), activation='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Conv2D(128, (2, 2), activation='relu'))
Discriminator.add(MaxPooling2D(pool_size=(2, 2)))

Discriminator.add(Flatten())
Discriminator.add(Dense(128, activation='relu'))
Discriminator.add(Dense(1, activation='sigmoid'))

print("===== DISCRIMINATOR =====")
Discriminator.summary()
Discriminator.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

###   CREER LE GENERATOR   ###

Generator = Sequential()
Generator.add(Dense(16*16*128, activation='relu', input_shape=(16,)))

Generator.add(Reshape((16, 16, 128)))
#Entrée : 16, 16, 128

Generator.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
Generator.add(UpSampling2D(size=(2, 2)))
#32*32
Generator.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
Generator.add(UpSampling2D(size=(2, 2)))
#64*64
Generator.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
Generator.add(UpSampling2D(size=(2, 2)))
#128*128
Generator.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))
#couche de sortie. 128, 128, 3
Generator.add(Conv2D(3, (2, 2),padding='same' ,activation='tanh'))

print()
print()
print("===== GENERATOR =====")
Generator.summary()

###   ON ENTRAINE TOUT CA  ###
# Je Créer le combo des deux #

Combo = Sequential()

Combo.add(Generator)
Combo.add(Discriminator)

Combo.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

BoucleActuelle = 0

while True :
    # ON ENTRAINE LE DISCRIMINATOR #

    print()
    print()
    print("===== Boucle N°" + str(BoucleActuelle) + " =====")
    Images_Vraies = Images[np.random.randint(0, Images.shape[0], size=Batch_Size)]
    Label_Vrais = np.ones(Batch_Size)

    Bruit = np.random.normal(0, 1, (Batch_Size, 16))
    Images_Fausses = Generator.predict(Bruit)
    Label_Faux = np.zeros(Batch_Size)

    Images_Entrainement = np.concatenate([Images_Vraies, Images_Fausses])
    Label_Entrainement = np.concatenate([Label_Vrais, Label_Faux])

    Discriminator.trainable = True
    print()
    print()
    print("===== ENTRAINEMENT DISCRIMINATEUR =====")
    print()
    print()
    Discriminator.fit(Images_Entrainement, Label_Entrainement, epochs=1, batch_size=32)

    # ON ENTRAINE LE COMBO DES DEUX #
    Bruit = np.random.normal(0, 1, (Batch_Size * 2, 16))
    Labels = np.ones(Batch_Size * 2)
    Discriminator.trainable = False
    print()
    print()
    print("===== ENTRAINEMENT COMBO =====")
    print()
    print()
    Combo.fit(Bruit, Labels, epochs=1, batch_size=16)

### ON SAVE SE QUIL Y A A SAVE ###

    if BoucleActuelle % 5 == 0:
        bruit = np.random.normal(0, 1, (1, 100))
        image = Generator.predict(bruit)
        image = image.reshape(16, 16, 3)
        image = image * 127.5 + 127.5
        image = image.astype("uint8")
        nomImage = "Image Cycle N°" + str(BoucleActuelle) + ".jpg"
        cv2.imwrite("doss/" + nomImage, image)
        BoucleActuelle += 1

        Generator.save("Generator.h5")
        Discriminator.save("Discriminator.h5")
