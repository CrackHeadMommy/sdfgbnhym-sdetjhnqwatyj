import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

bildes = []
label = []

bilzu_adrese = "majas/bildes/"

for nosaukums in os.listdir(bilzu_adrese):
    image = Image.open(os.path.join(bilzu_adrese,nosaukums))
    bildes.append(np.array(image))
    if "maja" in nosaukums:
        label.append(1)
    else:
        label.append(0)

X_train, X_test, Y_train, Y_test = train_test_split(bildes, label, test_size=0.2, random_state=0)

modelis = RandomForestClassifier()

modelis.fit(X_train, Y_train)

paregojums = modelis.predict(X_test)

precizitate = accuracy_score(Y_test, paregojums)

print(precizitate)
