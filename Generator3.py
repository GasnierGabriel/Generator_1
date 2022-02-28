## IMPORT ##
from glob import glob
from tqdm import tqdm
import cv2
from keras.layers import Dense, ReLU, MaxPooling2D, Conv2D, Flatten, UpSampling2D, Reshape, BatchNormalization, Dropout, Conv2DTranspose, Activation, LeakyReLU
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras
import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.9)
Batch = 8
dossier_Images = "cats0"
cycle = 1000000


## RECUP DATA ##

Images = []

nom_images = glob(dossier_Images + "/*.jpg")
for nom_de_limage_actuel_a_traiter in tqdm(nom_images):
    image = cv2.imread(nom_de_limage_actuel_a_traiter, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = (image.astype("float32") - 127.5) / 127.5
    Images.append(image)

Images = np.asarray(Images)

## DISCRIMINATEUR ##

Discriminator = Sequential()

Discriminator.add(Conv2D(1024, kernel_size=(4, 4),
                         activation ='relu',
                         input_shape=(128, 128, 3)))
Discriminator.add(MaxPooling2D(pool_size=(4, 4)))
Discriminator.add(LeakyReLU())
#Discriminator.add(Dropout(0.25))


Discriminator.add(Conv2D(1500, kernel_size=(4, 4),
                         activation ='relu'))
Discriminator.add(MaxPooling2D(pool_size=(4, 4)))
Discriminator.add(LeakyReLU())
#Discriminator.add(Dropout(0.25))

#Discriminator.add(Conv2D(1500, kernel_size=(4, 4),activation ='relu'))
#Discriminator.add(MaxPooling2D(pool_size=(4, 4)))
#Discriminator.add(LeakyReLU())
#Discriminator.add(Dropout(0.25))

Discriminator.add(Flatten())
Discriminator.add(LeakyReLU())
Discriminator.add(Dropout(0.5))
Discriminator.add(Dense(1, activation='sigmoid'))

print("===== DISCRIMINATOR =====")

Discriminator.summary()
Discriminator.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.5),
              metrics=['accuracy'])

## GENERATEUR ##

Generator = Sequential()

Generator.add(Dense(8*8*128, activation="relu", input_shape=(128,)))
Generator.add(Reshape((8, 8, 128)))
Generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
Generator.add(LeakyReLU())
#25x25x128
Generator.add(Conv2DTranspose(32))
Generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
Generator.add(LeakyReLU())
#50x50x64

Generator.add(Conv2D(1200, kernel_size=(2, 2), padding='same', activation="relu"))
Generator.add(UpSampling2D(size=(2, 2)))
Generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
Generator.add(LeakyReLU())

Generator.add(Conv2D(1100, kernel_size=(2, 2), padding='same', activation="relu"))
Generator.add(UpSampling2D(size=(2, 2)))
Generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
Generator.add(LeakyReLU())

Generator.add(Conv2D(1000, kernel_size=(2, 2), padding='same', activation="relu"))
Generator.add(UpSampling2D(size=(2, 2)))
Generator.add(BatchNormalization(momentum=0.1, epsilon=1e-05))
Generator.add(LeakyReLU())


#100x100x3
Generator.add(Conv2D(3, kernel_size=(2, 2), padding='same', activation='tanh'))


print()
print()
print("===== GENERATOR =====")

Generator.summary()
#Sortie 100 x 100 X 3

Combo = Sequential()

Combo.add(Generator)
Combo.add(Discriminator)

Combo.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

## ENTRAINEMENT DU DISCRIMINATEUR ##

CycleActuel = 0
while CycleActuel < cycle:
    #On récupère nos images des données.
    Batch_vraies = Images[np.random.randint(0, Images.shape[0], size=Batch)]
    Labels_vraies = np.ones(Batch)

    # On créer du bruit.
    Bruit = np.random.normal(0, 1, size=[Batch, 128])

    #On créer des images à partir du bruit.
    Images_Fausse = Generator.predict(Bruit)
    Labels_faux = np.zeros(Batch)

    Images_Entrainement = np.concatenate([Batch_vraies, Images_Fausse])
    Labels_Entrainement = np.concatenate([Labels_vraies, Labels_faux])

    Discriminator.trainable = True
    print()
    print("Cycle N°" + str(CycleActuel))
    print()
    print()
    print("====== Entrainement Discriminator =======")
    print()
    print()
    Discriminator.fit(Images_Entrainement, Labels_Entrainement, epochs=1, batch_size=100)

## ENTRAINEMENT DE L'AUTRE (Et non ! DU COMBO DES DEUX) ##
   ### GENERER DU BRUIT ###

    Bruit = np.random.normal(0, 1, size=[Batch, 128])
    Labels_Combo = np.ones(Batch)

    print()
    print("Cycle N°" + str(CycleActuel))
    print()
    print()
    print("====== Entrainement Combo =======")
    print()
    print()

    Discriminator.trainable = False
    Combo.fit(Bruit, Labels_Combo, epochs=1, batch_size=100)


## ON SAVE CE QU'IL Y A A SAUVER ##

    if CycleActuel % 5 == 0 :
        bruit = np.random.normal(0, 1, size=[1, 128])
        image = Generator.predict(bruit)
        image = (image * 127.5) + 127.5
        image = image.astype("uint8")
        image = np.reshape(image, (128, 128, 3))
        NomImage = "ImageGenere cycle N°" + str(CycleActuel) + ".jpg"
        cv2.imwrite("doss/" + NomImage, image)
        Generator.save("Generator.h5")
        Discriminator.save("Discriminator.h5")
    CycleActuel += 1

