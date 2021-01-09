import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from random import randint
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
n=40
m=6
w0=[1 for i in range(m)]
t=[]
yt=[]

#oluşturulan datasetin çekilmesi
p=np.load('dataset1.npy')
p=np.array(p,dtype=np.float64)

#perceptron algoritmasının fonksiyonu
def GAA(p,w0,c):
    train,test= train_test_split(p,train_size=0.625)
    train=np.array(train,dtype=np.float64)
    test=np.array(test,dtype=np.float64)
    ytr=[]
    k1=[]
    k=0
    iter=0
    w1=0
    #test ve eğitim kümesi için doğru yd değerlerinin bulunması
    for i in range (len(train)):
        if sum(train[i])>0:
            ytr.append(1)
        else:
            ytr.append(-1)
    ytr=np.array(ytr,dtype=np.float64)
    #algoritmanın başlangıcı
    while True:
        for i in range(len(train)):
            s=sum(train[i]*w0) #w0*x
            if s>0: #sınıf kontrolü
                y1=1
                d=((1/2)*(ytr[i]-y1)*train[i])
                w0=w0+(c*d) # ağırlık güncellenmesi
                if ytr[i]==1:
                    k+=1 # doğru sınıftaysa kyı arttır
            elif s<=0: #sınıf kontrolü
                y1=-1 
                d=((1/2)*(ytr[i]-y1)*train[i])
                w0=w0+(c*d) # ağırlık güncellenmesi
                if ytr[i]==-1:
                    k+=1 # doğru sınıftaysa kyı arttır
        iter+=1 #iterasyonu arttır.
        k1.append(k)
        if k==25: #eğitim kümesi tamamen doğru olduğunda durdur
            print("iterasyon sayısı",iter)
            break
        k=0
    yte=[]
    dogru=0
    #test kümesi için doğru yd değerleri
    for i in range (0,15):
        if sum(test[i])>0:
            yte.append(1)
        else:
                yte.append(-1)
    yte=np.array(yte,dtype=np.float64)
    #bulunan ağırlık ve test kümesi elemanlarının çarpılması
    for i in range(15):
        s=sum(test[i]*w0)
        if s>0:
            y1=1
        else:
            y1=-1
        yt.append(y1)
#test kümesindeki doğru sayısı hesaplanması
    for i in range(len(test)):
        if yt[i]-yte[i]==0:
            dogru+=1
    print("test kümesi dogru sayısı",dogru)


#dataset sabit,öğrenme hızı sabit,ilk ağırlıklar değişirken
print ("Ağırlıklar değişirken")
for i in range(20):
    top=0
    c=0.88
    for i in range(0,6):
        w0[i]=random.randint(-100000,100000)
    GAA(p,w0,c)

  
#dataset sabit,ilk ağırlıklar sabit,öğrenme hızı değişirken
print("öğrenme hızı değişirken") 
for i in range(20):
    w0=[100,100,-100,100,100,-100]
    c=random.uniform(0,1)
    GAA(p,w0,c)

#dataset karıştırılır,diğer değerler sabitken
print("dataset karıştırılır,diğer değerler sabitken")
for i in range(0,6):
    w0[i]=random.randint(-100000,100000)
c=random.uniform(0,1)
for i in range(20):
    np.random.shuffle(p)
    GAA(p,w0,c)

#çizim işlemleri
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(p[:,3],p[:,4],p[:,5],c='r')
plt.show()

