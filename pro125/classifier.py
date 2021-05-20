import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from PIL import Image 
import PIL.ImageOps

X  = np.load("image.npz")['arr_0']
y = pd.read_csv("Pro123 - Data.csv")["labels"]
print(pd.Series(y).value_counts())
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
n = len(labels)
print(n)

x_train , x_test , y_train , y_test = train_test_split(X , y , random_state = 9 , train_size = 7500 , test_size = 2500)
x_train_scl = x_train/255
x_test_scl = x_test/255

clf = LogisticRegression(solver="saga" , multi_class="multinomial" ).fit(x_train_scl , y_train)


#Predicting the image
def ImgPred(image):
    im_pil = Image.open(image)
    im_bw = im_pil.convert('L')
    im_rs = im_bw.resize((28 , 28) , Image.ANTIALIAS)
    px_fil = 20
    min_px = np.percentile(im_rs , px_fil)
    im_inv = np.clip(im_rs-min_px , 0 , 255)
    max_px = np.max(im_rs)
    im_inv = np.asarray(im_inv)/max_px
    test_sam = np.array(im_inv).reshape(1 , 784)
    test_pred = clf.predict(test_sam)
    return test_pred[0]


