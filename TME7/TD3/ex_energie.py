# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# poids synaptique r?current et l'entr?e du neurone
w = 10.
x = -w/2.

# fonction d'actvation
g = lambda a: 1./(1+np.exp(-a))

# fonction energie
E = lambda y: y**2./2. -y - np.log(1+np.exp(-w*y-x))/w

# param?tres de la simulation
dt = 0.1
time = np.arange(0,10,dt)
T = len(time)
y = np.zeros(T)

# condition initiale pour y(t)
y[0] = 0.4

# initialisation du graphisme
plt.figure(1); plt.show(); plt.clf()

y_interv = np.arange( -0.5, 1.5, 0.01)

for t in range(1, T):
    
    # m?thode d'Euler pour dy/dt=-y+g(wy+x)
    y[t] = y[t-1]+ dt*(-y[t-1] + g(w*y[t-1]+x)) 

    # fonction energie
    plt.subplot(211); plt.cla()
    plt.plot(y_interv, E(y_interv), 'k-', lw=3 )
    plt.plot( [y[t]], [E(y[t])], 'ro', markersize=10)
    plt.axhline(0, ls=':', color = 'k')
    plt.axvline(0, ls=':', color = 'k')
    plt.draw()
    
    # solution
    plt.subplot(212); plt.cla()
    plt.plot(time[:t+1], y[:t+1], lw=3 )
    plt.axhline(0, ls='-', color = 'k', lw=3)
    plt.axvline(0, ls='-', color = 'k', lw=3)
    plt.xlim(-1,8)
    plt.ylim(-0.1,1.1)
    plt.draw()
    
    sleep(0.1)

