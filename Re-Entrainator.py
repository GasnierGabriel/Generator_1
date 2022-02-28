###  IMPORT  ###
from glob import glob
from tqdm import tqdm
import cv2
import keras
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np


### PARAMETRES ###
optimizer = Adam(lr=0.0002, beta_1=0.5)
Batch_Size= 2000
NombreBoucle = 500

### RECUP DATA ###

Images = []

dossier_images = glob("im/*")
for dossier in dossier_images:
    nom_images = glob(dossier + "/*.jpg")
    for nom_de_limage_actuel_a_traiter in tqdm(nom_images):
        image = cv2.imread(nom_de_limage_actuel_a_traiter, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))
        image = (image.astype("float32") - 127.5) / 127.5
        Images.append(image)

Images = np.array(Images)

### CREER LE DISCRIMINATOR ###
Discriminator = keras.models.load_model("Discriminator.h5")

Generator = keras.models.load_model("Generator.h5")

###   ON ENTRAINE TOUT CA  ###
# Je Créer le combo des deux #

Combo = Sequential()

Combo.add(Generator)
Combo.add(Discriminator)

Combo.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

BoucleActuelle = 350

while True :
    # ON ENTRAINE LE DISCRIMINATOR #

    print()
    print()
    print("===== Boucle N°" + str(BoucleActuelle) + " =====")
    Images_Vraies = Images[np.random.randint(0, Images.shape[0], size=Batch_Size)]
    Label_Vrais = np.ones(Batch_Size)

    Bruit = np.random.normal(0, 1, (Batch_Size, 100))
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
    Bruit = np.random.normal(0, 1, (Batch_Size, 100))
    Labels = np.ones(Batch_Size)
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
        image = image.reshape(128, 128, 3)
        image = image * 127.5 + 127.5
        image = image.astype("uint8")
        nomImage = "Image Cycle N°" + str(BoucleActuelle) + ".jpg"
        cv2.imwrite("DossierTest/" + nomImage, image)

    BoucleActuelle += 1

    Generator.save("Generator.h5")
    Discriminator.save("Discriminator.h5")