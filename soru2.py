#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import math


# In[2]:


#Scatter fonksiyonunu gözlemleme
X=np.random.rand(100,2)
for i in range (len (X)):
    if (7*X[i,0]-2)<5*X[i,1]:
        plt.scatter(X[i,0],X[i,1],c='b')
    else:
        plt.scatter(X[i,0],X[i,1],c='r')


# In[3]:


#Elimizdeki datayı scatter fonksiyonu ile yazdırma
#Ve lineer ayrışmazlığı gözlemleme
set1=np.array([[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1],[-1,-1],[-1,0],[-1,1]])
set2=np.array([[-3,3],[-3,1],[-3,0],[-3,-1],[-3,-3],[-1,3],[-1,-3],
        [0,3],[0,-3],[1,3],[1,-3],[3,3],[3,1],[3,0],[3,-1],
        [3,-3],[-2,3],[-3,2],[-3,-2],[-2,-3],[2,3],[3,2],[3,-2],[2,-3]])
plt.scatter(set1[:,0],set1[:,1],c='r')
plt.scatter(set2[:,0],set2[:,1],c='g')     
plt.show()        


# In[4]:


#3 boyutlu düzlemde scatter fonkiyonu kullanımını gözlemleme
fig = plt.figure()
ax = Axes3D(fig)
set1=np.array([[0,-1,1],[0,0,1],[0,1,1],[1,-1,1],[1,0,1],[1,1,1],[-1,-1,1],[-1,0,1],[-1,1,1]])
set2=np.array([[-3,3],[-3,1],[-3,0],[-3,-1],[-3,-3],[-1,3],[-1,-3],
        [0,3],[0,-3],[1,3],[1,-3],[3,3],[3,1],[3,0],[3,-1],
        [3,-3],[-2,3],[-3,2],[-3,-2],[-2,-3],[2,3],[3,2],[3,-2],[2,-3]])
ax.scatter(set1[:,0],set1[:,1],set1[:,2],c='r')
ax.scatter(set2[:,0],set2[:,1],c='g')     
plt.show()    


# In[35]:


#Kullanılan kütüphaneler
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import math
from numpy import random
#bias eklenmiş veri seti
ds=([[0,-1,1],[0,0,1],[0,1,1],[1,-1,1],[1,0,1],[1,1,1],[-1,-1,1],[-1,0,1],[-1,1,1],
        [-3,3,1],[-3,1,1],[-3,0,1],[-3,-1,1],[-3,-3,1],[-1,3,1],[-1,-3,1],
        [0,3,1],[0,-3,1],[1,3,1],[1,-3,1],[3,3,1],[3,1,1],[3,0,1],[3,-1,1],
        [3,-3,1],[-2,3,1],[-3,2,1],[-3,-2,1],[-2,-3,1],[2,3,1],[3,2,1],[3,-2,1],[2,-3,1]])
#veri setinin doğru yanıtları
y=([1],[1],[1],[1],[1],[1],[1],[1],[1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1])
w0=[1,1,1,1]
w0=np.array(w0)
c=0.5
bias=1
hata=0
iter=0
k=0
k1=[]
yt=[]
#3.boyutu 1 olarak eklemek için,
for i in range(33):
    ds[i].append(1)
#bias'i sonradan eklenmiş olan 3.boyuta taşıma
ds=np.array(ds)
for i in range(33):
    ds[i][3]=ds[i][2]           

ds=np.array(ds,dtype=np.float64)
#lineer ayrıştırılabilir hale getirildi
#3.boyutu belirme
for i in range(33):
    a= (ds[i][0]*ds[i][0]+3)+(ds[i][1]*ds[i][1])
    ds[i][2]= 2/a

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(ds[:,0],ds[:,1],ds[:,2],c='r')
plt.show()

#train ve test kümeleri ayrımları
train=(ds[5],ds[6],ds[7],ds[8],ds[9],ds[10],ds[11],ds[12],ds[13],ds[14],ds[15],ds[16],ds[17],ds[18],ds[19])
train=np.array(train,dtype=np.float64)
test=(ds[1],ds[2],ds[3],ds[4],ds[20],ds[21],ds[22],ds[23],ds[24],ds[25],ds[26],ds[27],ds[28],ds[29],ds[30],ds[31],ds[32])
test=np.array(test,dtype=np.float64)
print(test) #test kümesini yazdırma

ytr=(y[5],y[6],y[7],y[8],y[9],y[10],y[11],y[12],y[13],y[14],y[15],y[16],y[17],y[18],y[19])
yte=(y[1],y[2],y[3],y[4],y[20],y[21],y[22],y[23],y[24],y[25],y[26],y[27],y[28],y[29],y[30],y[31],y[32])
ytr=np.array(ytr)
yte=np.array(yte)

#perceptron algoritması
while iter<100:
    for i in range(len(train)):
        s=sum(train[i]*w0)
        if s>0.2: #sınıf kontrol edildi
            y1=1
            d=((1/2)*(ytr[i]-y1)*train[i])
            w0=w0+(c*d) #w0 güncelleme
            if ytr[i]==1:
                k+=1
        elif s<=0.2: #sınıf kontrol edildi
            y1=-1 
            d=((1/2)*(ytr[i]-y1)*train[i])
            w0=w0+(c*d) #w0 güncelleme
            if ytr[i]==-1:
                k+=1
    iter+=1
    k1.append(k)
    print("iterasyon sayısı:",iter,"doğru sayısı:",k)

    if k==15:
        break #öğrenme bittiği zaman döngüden çıkış
    k=0
#test
dogru=0
for i in range(len(test)):
    s=sum(test[i]*w0)
    if s>0:
        y1=1
    else:
        y1=-1
    yt.append(y1)

for i in range(len(test)):
    if yt[i]-yte[i]==0:
        dogru+=1

print("test kümesi doğru sayısı",dogru)

