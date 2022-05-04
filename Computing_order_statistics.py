# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 18:13:26 2021

@author: Ricardo
"""
import numpy as np
from numpy.random import seed
from numpy.random import normal
import math
import scipy.special
#import lmom
from scipy.stats import uniform 
import seaborn as sns
import random
import matplotlib.pyplot as plt

from math import factorial



#make this example reproducible
#seed(4)

#generate sample of 200 values that follow a normal distribution 
data = normal(loc=0, scale=1, size=250)
data=np.float32(data)

#view first six values
#ordered Statistics

data=np.sort(data)

n=data.shape[0]

#number of moments:
k=5
a=np.zeros([k],dtype=float)
l=np.zeros([k],dtype=float)
a[0]=sum(data)/n
l[0]=a[0]

"""
for r in range(1,k):
    s=0
    for j in range(r+1,n+1):
        numerator=math.factorial(j-1)/math.factorial(j-r-1)
        denominator=math.factorial(n-1)/math.factorial(n-r-1)
        s=numerator/denominator*data[j-1]+s
    a[r]=(1/n)*s


for r in range(0,k):
    s=0
    for j in range(0,r):
        (-1)**(r-j)
        r_C_j=math.factorial(r)/(math.factorial(j)*math.factorial(r-j))
        rj_C_j=math.factorial(r+j)/(math.factorial(j)*math.factorial(r))
        s=(-1)**(r-j)*rj_C_j*r_C_j*a[j]+s
    l[r]=((-1)**r)*s
"""
var=lmom.samlmu(data,5)


n=4
unif1=np.zeros([n],dtype=int)
for I in range(0,n):
    I=random.randint(0, n-1)
    unif1[I]=unif1[I]+1
unif2=np.zeros([n],dtype=int)
for I in range(0,n):
    I=random.randint(0, n-1)
    unif2[I]=unif2[I]+1
unif3=np.zeros([n],dtype=int)
for I in range(0,n):
    I=random.randint(0, n-1)
    unif3[I]=unif3[I]+1

unif1=unif1
unif2=unif2
unif3=unif3

sss=unif1+unif2+unif3



L=0
M=1000
#unif3=1/M*np.ones([n],dtype=float)


n=3
Dists=set()
#Dists=[]
D=0
for J in range(0,M):
    unif1=np.zeros([n],dtype=int)
    for i in range(0,3*n):
        I=random.randint(0,n-1)
        unif1[I]=unif1[I]+1
        v=str(unif1)
    if v not in Dists:
        Dists.add(v)
        D=D+1


L=0
M=1000
for J in range(0,M):
    unif1=np.zeros([n],dtype=int)
    for i in range(0,3*n):
        I=random.randint(0,n-1)
        unif1[I]=unif1[I]+1
        v=str(unif1)
    if v=='[0 3 6]':
        L=L+1
        print(L)



M=10000
count=0
store=np.zeros([len(Dists)],dtype=float)
arr=np.array(Dists)
P=0
for K in enumerate(Dists):
    L=0
    for J in range(0,M):
        unif1=np.zeros([n],dtype=int)
        for i in range(0,3*n):
            I=random.randint(0,n-1)
            unif1[I]=unif1[I]+1
            v=str(unif1)
        if v==K[1]:
            L=L+1
    count=count+1
    print(K[1],L/M)
    store[P]=L
    P=P+1


def nPr(n, r):
  return int(factorial(n)/factorial(n-r))

def mult(n, i1,i2,i3):
  return int(factorial(n)/(factorial(i1)*factorial(i2)*factorial(i3)))


ss=0
for I in range(0,3*n+1):
    for J in range(0,3*n+1):
        for K in range(0,3*n+1):
            if I+J+K==3*n:
                if I==0 or J==0 or K==0:
                    ss=ss+mult(3*n,I,J,K)/n**(3*n)
            else:
                ss=ss




def mult4(n, i1,i2,i3,i4):
  return int(factorial(n)/(factorial(i1)*factorial(i2)*factorial(i3)*factorial(i4)))

def mult5(n, i1,i2,i3,i4,i5):
  return int(factorial(n)/(factorial(i1)*factorial(i2)*factorial(i3)*factorial(i4)*factorial(i5)))

def mult6(n, i1,i2,i3,i4,i5,i6):
  return int(factorial(n)/(factorial(i1)*factorial(i2)*factorial(i3)*factorial(i4)*factorial(i5)*factorial(i6)))

def mult7(n, i1,i2,i3,i4,i5,i6,i7):
  return int(factorial(n)/(factorial(i1)*factorial(i2)*factorial(i3)*factorial(i4)*factorial(i5)*factorial(i6)*factorial(i7)))

def mult8(n, i1,i2,i3,i4,i5,i6,i7,i8):
  return int(factorial(n)/(factorial(i1)*factorial(i2)*factorial(i3)*factorial(i4)*factorial(i5)*factorial(i6)*factorial(i7)*factorial(i8)))

def mult9(n, i1,i2,i3,i4,i5,i6,i7,i8,i9):
  return int(factorial(n)/(factorial(i1)*factorial(i2)*factorial(i3)*factorial(i4)*factorial(i5)*factorial(i6)*factorial(i7)*factorial(i8)*factorial(i9)))



ss=0
n=4
for I in range(0,3*n+1):
    for J in range(0,3*n+1):
        for K in range(0,3*n+1):
            for I1 in range(0,3*n+1):
                if I+J+K+I1==3*n:
                    if I==0 or J==0 or K==0 or I1==0:
                        ss=ss+mult4(3*n,I,J,K,I1)/(n**(3*n))
                    else:
                        ss=ss


ss=0
n=5
for I in range(0,3*n+1):
    for J in range(0,3*n+1):
        for K in range(0,3*n+1):
            for I1 in range(0,3*n+1):
                for I2 in range(0,3*n+1):
                    if I+J+K+I1+I2==3*n:
                        if I==0 or J==0 or K==0 or I1==0 or I2==0:
                            ss=ss+mult5(3*n,I,J,K,I1,I2)/(n**(3*n))
                        else:
                            ss=ss

ss=0
n=6
for I in range(0,3*n+1):
    for J in range(0,3*n+1):
        for K in range(0,3*n+1):
            for I1 in range(0,3*n+1):
                for I2 in range(0,3*n+1):
                    for I3 in range(0,3*n+1):
                        if I+J+K+I1+I2+I3==3*n:
                            if I==0 or J==0 or K==0 or I1==0 or I2==0 or I3==0:
                                ss=ss+mult6(3*n,I,J,K,I1,I2,I3)/(n**(3*n))
                            else:
                                ss=ss


ss=0
n=7
for I in range(0,3*n+1):
    for J in range(0,3*n+1):
        for K in range(0,3*n+1):
            for I1 in range(0,3*n+1):
                for I2 in range(0,3*n+1):
                    for I3 in range(0,3*n+1):
                        for J1 in range(0,3*n+1):
                            if I+J+K+I1+I2+I3+J1==3*n:
                                if I==0 or J==0 or K==0 or I1==0 or I2==0 or I3==0 or J1==0:
                                    ss=ss+mult7(3*n,I,J,K,I1,I2,I3,J1)
                                else:
                                    ss=ss
ss=ss/(n**(3*n))



ss=0
n=8
for I in range(0,3*n+1):
    for J in range(0,3*n+1-I):
        for K in range(0,3*n+1-(I+J)):
            for I1 in range(0,3*n+1-(I+J+K)):
                for I2 in range(0,3*n+1-(I+J+K+I1)):
                    for I3 in range(0,3*n+1-(I+J+K+I1+I2)):
                        for J1 in range(0,3*n+1-(I+J+K+I1+I2+I3)):
                            for J2 in range(0,3*n+1-(I+J+K+I1+I2+I3+J1)):
                                if I+J+K+I1+I2+I3+J1+J2==3*n:
                                    if I==0 or J==0 or K==0 or I1==0 or I2==0 or I3==0 or J1==0 or J2==0:
                                        ss=ss+mult8(3*n,I,J,K,I1,I2,I3,J1,J2)
                                    else:
                                        ss=ss
ss=ss/(n**(3*n))


ss=0
n=9
for I in range(0,3*n+1):
    for J in range(0,3*n+1-I):
        for K in range(0,3*n+1-(I+J)):
            for I1 in range(0,3*n+1-(I+J+K)):
                for I2 in range(0,3*n+1-(I+J+K+I1)):
                    for I3 in range(0,3*n+1-(I+J+K+I1+I2)):
                        for J1 in range(0,3*n+1-(I+J+K+I1+I2+I3)):
                            for J2 in range(0,3*n+1-(I+J+K+I1+I2+I3+J1)):
                                for J3 in range(0,3*n+1-(I+J+K+I1+I2+I3+J1+J2)):
                                    if I+J+K+I1+I2+I3+J1+J2+J3==3*n:
                                        if I==0 or J==0 or K==0 or I1==0 or I2==0 or I3==0 or J1==0 or J2==0 or J3==0:
                                            ss=ss+mult9(3*n,I,J,K,I1,I2,I3,J1,J2,J3)
                                        else:
                                            ss=ss
ss=ss/(n**(3*n))



#THIS IS THE ANSWER

ss=0
n=9
for I in range(0,3*n):
    for J in range(0,3*n-I):
        for K in range(0,3*n-(I+J)):
            for I1 in range(0,3*n-(I+J+K)):
                for I2 in range(0,3*n-(I+J+K+I1)):
                    for I3 in range(0,3*n-(I+J+K+I1+I2)):
                        for J1 in range(0,3*n-(I+J+K+I1+I2+I3)):
                            for J2 in range(0,3*n-(I+J+K+I1+I2+I3+J1)):
                                for J3 in range(0,3*n-(I+J+K+I1+I2+I3+J1+J2)):
                                    if I+J+K+I1+I2+I3+J1+J2+J3==3*n-1:
                                        if I==0 or J==0 or K==0 or I1==0 or I2==0 or I3==0 or J1==0 or J2==0 or J3==0:
                                            ss=ss+mult9(3*n,I,J,K,I1,I2,I3,J1,J2,J3)
                                        else:
                                            ss=ss
#ss9=ss/(n**(3*n-1)+1)*1/n


c=0
for I in range(0,9):
    for J in range(0,9):
        for K in range(0,9):
            if I+J+K==8:
                c=c+mult(8,I,J,K)

c=0
n=9
for I in range(0,3*n):
    for J in range(0,3*n-I):
        for K in range(0,3*n-(I+J)):
            for I1 in range(0,3*n-(I+J+K)):
                for I2 in range(0,3*n-(I+J+K+I1)):
                    for I3 in range(0,3*n-(I+J+K+I1+I2)):
                        for J1 in range(0,3*n-(I+J+K+I1+I2+I3)):
                            for J2 in range(0,3*n-(I+J+K+I1+I2+I3+J1)):
                                for J3 in range(0,3*n-(I+J+K+I1+I2+I3+J1+J2)):
                                    if I+J+K+I1+I2+I3+J1+J2+J3==3*n-1:
                                        c=c+mult9(3*n-1,I,J,K,I1,I2,I3,J1,J2,J3)
                                    else:
                                        c=c

answer=ss/(c*3*n)


"""
n=3
for J in range(0,M):
    unif1=np.zeros([n],dtype=int)
#    for K in range(0,3):
#        for I in range(0,n):
#            I=random.randint(0,n-1)
#            unif1[I]=unif1[I]+1

#    for i in range(0,3*n-1):
    for i in range(0,n):
        I=random.randint(0,n-1)
        unif1[I]=unif1[I]+1

#    sss=unif1
    if min(unif1)==0:
        L=L+1

print(L/M)
"""


print(2*(1/2)**6)
print(3*(1/3)**9+3*(2/3)**9)
print(4*(1/4)**12+6*(2/4)**12+4*(3/4)**12)
print(5*(1/5)**15+10*(2/5)**15+10*(3/5)**15+5*(4/5)**15)
print(6*(1/6)**18+15*(2/6)**18+20*(3/6)**18+15*(4/6)**18+6*(5/6)**18)
print(7*(1/7)**21+21*(2/7)**21+35*(3/7)**21+35*(4/7)**21+21*(5/7)**21+7*(6/7)**21)
print(8*(1/8)**24+28*(2/8)**24+56*(3/8)**24+70*(4/8)**24+56*(5/8)**24+28*(6/8)**24+8*(7/8)**24)
print(9*(1/9)**27+36*(2/9)**27+84*(3/9)**27+126*(4/9)**27+126*(5/9)**27+84*(6/9)**27+36*(7/9)**27+9*(8/9)**27)


print(2*(1/2)**5)
print(3*(1/3)**8+3*(2/3)**8)
print(4*(1/4)**11+6*(2/4)**11+4*(3/4)**11)
print(5*(1/5)**14+10*(2/5)**14+10*(3/5)**14+5*(4/5)**14)
print(6*(1/6)**17+15*(2/6)**17+20*(3/6)**17+15*(4/6)**17+6*(5/6)**17)
print(7*(1/7)**20+21*(2/7)**20+35*(3/7)**20+35*(4/7)**20+21*(5/7)**20+7*(6/7)**20)
print(8*(1/8)**23+28*(2/8)**23+56*(3/8)**23+70*(4/8)**23+56*(5/8)**23+28*(6/8)**23+8*(7/8)**23)
print(9*(1/9)**26+36*(2/9)**26+84*(3/9)**26+126*(4/9)**26+126*(5/9)**26+84*(6/9)**26+36*(7/9)**26+9*(8/9)**26)



#print(2*(1/2)**5)
#print(1/3*3*(1/3)**8+3*1/3*(2/3)**8)
#print(4*3/4*(1/4)**11+6*3/4*(2/4)**11+4*3/4*(3/4)**11)
#print(4/5*5*(1/5)**14+4/5*10*(2/5)**14+4/5*10*(3/5)**14+4/5*5*(4/5)**14)
#print(5/6*6*(1/6)**17+5/6*15*(2/6)**17+5/6*20*(3/6)**17+5/6*15*(4/6)**17+5/6*6*(5/6)**17)
#print(6/7*7*(1/7)**20+6/7*21*(2/7)**20+6/7*35*(3/7)**20+6/7*35*(4/7)**20+6/7*21*(5/7)**20+6/7*7*(6/7)**20)
#print(7/8*8*(1/8)**23+7/8*28*(2/8)**23+7/8*56*(3/8)**23+7/8*70*(4/8)**23+1/8*56*(7/8)**23+1/8*28*(7/8)**23+1/8*8*(7/8)**23)
#print(1/9*9*(1/9)**26+1/9*36*(2/9)**26+1/9*84*(3/9)**26+1/9*126*(4/9)**26+1/9*126*(5/9)**26+1/9*84*(6/9)**26+1/9*36*(7/9)**26+1/9*9*(8/9)**26)






n=3
def binom(n, k):
    return math.comb(n, k)

def cut(n, k):
    return math.comb(n, k)

num_dist=(binom(3*n+n-1,3*n))


s=0
#for I in range(0,num_dist):
##    s=s+binom(3*n,I)*((1+I)/n)**(3*n-I)*((n-I)/n)**I
#    s=s+binom(3*n,I)*(1/n)**(3*n-I)*(1/n)**I
for I in range(0,num_dist):
    for I1 in range(0,n):
        for I2 in range(0,n):
            for I3 in range(0,n):
                s=s+binom(3*n,I)*(1/n)**I1*(1/n)**I2*(1/n)**I3
    




    





L=0
M=10000000
n=9
for J in range(0,M):

    unif1=np.zeros([n],dtype=int)

    for i in range(0,3*n-1):
        I=random.randint(0,n-1)
        unif1[I]=unif1[I]+1
    if min(unif1)==0:
        L=L+1

print(L/M)











"""
n = 10
start = 0
width = n
data_uniform1 = uniform.rvs(size=n, loc = start, scale=1)
data_uniform2 = uniform.rvs(size=n, loc = start, scale=1)
data_uniform3 = uniform.rvs(size=n, loc = start, scale=1)
summ=(data_uniform1+data_uniform2+data_uniform3)/3
"""

#Lin=np.linspace(0,n,n)
#plt.plot(Lin,sss)


#ax = sns.distplot(unif1,
#                  bins=n)
#                  kde=True,
#                  color='skyblue',
#                  hist_kws={"linewidth": 15,'alpha':1})
#ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')

"""
ax = sns.distplot(summ,
#ax = sns.distplot(data_uniform1,
                  bins=n,
                  kde=True,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Uniform Distribution ', ylabel='Frequency')
"""






ss=0
n=8
for I in range(0,3*n):
    for J in range(0,3*n-I):
        for K in range(0,3*n-(I+J)):
            for I1 in range(0,3*n-(I+J+K)):
                for I2 in range(0,3*n-(I+J+K+I1)):
                    for I3 in range(0,3*n-(I+J+K+I1+I2)):
                        for J1 in range(0,3*n-(I+J+K+I1+I2+I3)):
                            for J2 in range(0,3*n-(I+J+K+I1+I2+I3+J1)):
                                 if I+J+K+I1+I2+I3+J1+J2+J3==3*n-1:
                                    if I==0 or J==0 or K==0 or I1==0 or I2==0 or I3==0 or J1==0 or J2==0:
                                        ss=ss+mult8(3*n,I,J,K,I1,I2,I3,J1,J2)
                                    else:
                                        ss=ss






c=0
n=8
for I in range(0,3*n):
    for J in range(0,3*n-I):
        for K in range(0,3*n-(I+J)):
            for I1 in range(0,3*n-(I+J+K)):
                for I2 in range(0,3*n-(I+J+K+I1)):
                    for I3 in range(0,3*n-(I+J+K+I1+I2)):
                        for J1 in range(0,3*n-(I+J+K+I1+I2+I3)):
                            for J2 in range(0,3*n-(I+J+K+I1+I2+I3+J1)):
                                if I+J+K+I1+I2+I3+J1+J2==3*n-1:
                                        c=c+mult8(3*n-1,I,J,K,I1,I2,I3,J1,J2)
                                else:
                                    c=c

answer8=ss/(c*3*n)







from math import comb
import matplotlib.pyplot as plt

q = lambda N: sum(comb(N, j)*(1 - j/N)**(3*N-1)*(-1)**j for j in range(N))

a = lambda N: sum(comb(N, j)*(N - j)**(3*N-1)*(-1)**j for j in range(N))
C = lambda N: (2.0/3)*(N)**(3*N-1)

p = lambda N: 1 - q(N)


N = min([n for n in range(3,50) if p(n) >= 1.0/3])
N, p(N)


