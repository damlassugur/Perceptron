import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from random import randint
from sklearn.model_selection import train_test_split
import math
xtrain=np.load('dataset3.npy')
xtest=np.load('dataset4.npy')
E=[]
E1=[]
w=[]
lr=0.5
yd=[0 for i in range(len(xtrain))]
ydt=[0 for i in range(len(xtest))]
#yd değerlerinin fonksiyondan elle edilmesi
for i in range(len(xtrain)):
    yd[i]=3*xtrain[i][0]+2*math.cos(xtrain[i][1])
for i in range(len(xtest)):
    ydt[i]=3*xtest[i][0]+2*math.cos(xtest[i][1])
xtest=np.array(xtest)
xtrain=np.array(xtrain)
yd=np.array(yd)
#yd değerlerinin 0-1 arasında standardize edilmesi
for i in range (len(yd)):
    yd[i]=(yd[i]-min(yd))/(max(yd)-min(yd))

for i in range (len(ydt)):
    ydt[i]=(ydt[i]-min(ydt))/(max(ydt)-min(ydt))

#ilk ağırlıklar
w=[0.5,0.5,0.5]

w=np.array(w)
xtrain=np.array(xtrain)
a=0.4
iter=0
Ee=[]
#adaline algoritması 
while True:
    Et=[]
    for i in range(len(xtrain)):
        v=sum(w*xtrain[i]) #v=w*x
        y=1/(1+math.exp(-a*v)) #y=fi(v)
        fit=a*math.exp(-a*v)/(math.exp(-a*v)+1)*(math.exp(-a*v)+1) #fi'(v)
        e=yd[i]-y #hata 
        d=-((yd[i]-y))*fit*xtrain[i]
        Et.append(e) 
        w-=d*lr #ağırlık güncellemesi
    Et=np.array(Et)
    Ee.append(sum(Et*Et)/len(xtrain)) 
    if ((sum(Et*Et)/len(xtrain)))<1e-3: #toplam hata ve durdurma kriteri
        break
    iter+=1
    if iter>2000:
        break

print("öğrenme kümesi iterasyon sayısı:",iter,"Öğrenme kümesi Hata:",Ee)

plt.plot(Ee)
plt.show()
#test kümesi için ;
for i in range(len(xtest)):
    v=sum(w*xtest[i]) #v=w*x
    y=1/(1+math.exp(-a*v)) #y=fi(v)
    for i in range(len(xtest)): 
        E.append(ydt[i]-y)
    E1.append(sum(E)/(len(xtest)))
E=np.array(E)

print("test kümesinde hata:",E)
print("test kümesi MSE:",(sum(E)/(len(xtest))))


plt.plot(E)
plt.show()
plt.subplot(211)
plt.plot(yd)
plt.subplot(212)
plt.plot(w*xtest)
plt.show()