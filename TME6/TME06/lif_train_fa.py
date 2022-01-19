# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt
    
T   = 1000.                 # ms, Temps total de la simulation
dt  = 0.1                   # ms, Pas de temps de la simulation
t   = np.arange(0,T,dt)     # ms, Vecteur des temps en lesquels sont calcul�es les variables
N   = len(t)                # Longueur du vecteur des temps 

# Stimulation pr�-synaptique

f_pre    = np.arange(10, 1000, 10)       # Hz, Fr�quence des PA pr�-synaptiques

# Canaux r�cepteurs AMPA

tau_ampa = 5.26 # ms, Constante de temps de fermeture des canaux AMPA
p_max    = 1.   # Incr�ment d'ouverture des canaux lors d'un PA pr�-synaptique
g_ampa   = 0.1  # mS/cm2, conductance maximale de recepteurs AMPA
E_ampa   = 60.  # mV, conductance maximale de recepteurs AMPA

# Excitabilit�

tau_m =  10.  # ms, Constante de temps membranaire
r_m   =  10.  # KOhm.cm2, R�sistance de fuite 
theta =  20.  # mV, Seuil du potentiel d'action
v_P   =  100. # mV, Potentiel au pic du PA 

# Changement de la fr�quence pr�-synaptique
compte_pre   = np.zeros( len(f_pre))
compte_post  = np.zeros( len(f_pre))

for k in range( len(f_pre) ):

    # train des PA pr�-synaptiques
    p_pre  = f_pre[k]*dt/1000.         # Probabilit� d'avoir un PA dans un pas de temps
    pre    = np.random.rand(N)<p_pre   # Vecteur des PA pr�-synaptiques (0=rien 1=PA)

    # Initialisation

    i_ampa = np.zeros(N)
    p_ampa = np.zeros(N)
    v      = np.zeros(N)

    # Simulation

    n_spikes = 0
    for i in range(1,N):

        p_ampa[i] = p_ampa[i-1] - dt / tau_ampa * p_ampa[i-1] 
        
        if pre[i] == 1:
            p_ampa[i] = p_max
    
        # courant AMPA
        i_ampa[i] = g_ampa * p_ampa[i] * (v[i-1] - E_ampa) 
    
        # potentiel membranaire
        v[i] = v[i-1] + dt / tau_m * ( - v[i-1] - r_m * i_ampa[i] )
    
        # detection de PA
        if v[i-1] > theta:
            v[i] = v_P
    
        # reset � 0
        if v[i-1] == v_P:
            v[i] = 0
            n_spikes = n_spikes + 1

    compte_pre[k] = np.sum(pre)
    compte_post[k] = n_spikes

plt.figure(1); plt.show(); plt.clf()

# Figure PA pre-synaptique
plt.subplot(111) 
plt.plot(compte_pre,compte_post,'o')
plt.xlabel('Frequence pre, Hz')
plt.ylabel('Frequence post, Hz')
plt.draw()


