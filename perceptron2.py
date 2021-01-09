import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import random
from random import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
n=40
m=6
w0=[1 for i in range(m)]
yte=[]
ytr=[]
t=[]
yt=[]
k=0
k1=[]
iter=0
dogru=0

p=np.load('dataset5.npy')

#test ve train setlerinin bölünmesi
train,test= train_test_split(p,train_size=0.625)
#y değerleri
for i in range (0,25):
    if sum(train[i])>0:
        ytr.append(1)
    else:
        ytr.append(-1)
for i in range (0,15):
    if sum(test[i])>0:
        yte.append(1)
    else:
        yte.append(-1)

#pozitif w0 değeri
for i in range(0,6):
    w0[i]=random.randint(-1000000,10000)

w0=np.array(w0,dtype=np.float64)
train=np.array(train,dtype=np.float64)
test=np.array(test,dtype=np.float64)
yte=np.array(yte,dtype=np.float64)
ytr=np.array(ytr,dtype=np.float64)
p=np.array(p,dtype=np.float64)
#öğrenme hızının random atanması
c=random.uniform(0,1)

#perceptron algoritması
while iter<100:
    for i in range(0,25):
        s=sum(train[i]*w0) #x*wo
        if s>0:
            y1=1
            d=((1/2)*(ytr[i]-y1)*train[i])
            w0=w0+(c*d) #ağırlık güncellemesi
            if ytr[i]==1:
                k+=1 #doğru sayısını arttırır
        elif s<=0: 
            y1=-1 
            d=((1/2)*(ytr[i]-y1)*train[i])
            w0=w0+(c*d) #ağırlık güncelleme
            if ytr[i]==-1:
                k+=1 #doğru sayısını arttırma
    iter+=1 #iterasyon arttırma
    k1.append(k)
    print("iterasyon sayısı:",iter,"doğru sayısı:",k)

    if k==25: #eğitim kümesi tamamen doğru olduğunda eğitimi bitirme
        break
    k=0
#test kümesi için y değerlerinin bulunması
for i in range(15):
    s=sum(test[i]*w0)
    if s>0:
        y1=1
    else:
        y1=-1
    yt.append(y1)

#test kümesinde toplam doğru sayısının hesaplanması
for i in range(len(test)):
    if yt[i]-yte[i]==0:
        dogru+=1
    

plt.plot(k1)
plt.show()
print("dogru sayısı",dogru)

#çizim yapımı
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(p[:,2],p[:,3],p[:,5],c='r')
plt.show()

