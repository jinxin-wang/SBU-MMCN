# -*-coding:Latin-1 -*
import numpy as np
import matplotlib.pyplot as plt

T   = 1000                  # ms, Temps total de la simulation
dt  = 0.1                   # ms, Pas de temps de la simulation
t   = np.arange(0,T,dt)     # ms, Vecteur des temps en lesquels sont calcul�es les variables
N   = len(t)                # Longueur du vecteur des temps 

# Stimulation par un cr�neau de courant inject�

i_amp = np.arange(0, 0.02, 0.001)   # uA, Courant inject� pendant le cr�neau
t_on  = 0                           # ms, D�but du cr�neau
t_off = 1000                        # ms, Fin du cr�neau

# Excitabilit�

tau_m = 10    # ms, Constante de temps membranaire
R_m   = 4000  # KOhm, R�sistance de membrane 
theta = 20    # mV, Seuil du potentiel d'action
v_P   = 100   # mV, Potentiel au pic du PA 

# Changement de l'amplitude du courant
compte_pa  = np.zeros( len(i_amp) )

for k in range( len(i_amp) ):

    # Initialisation
    i_inj  = np.zeros(N)
    v      = np.zeros(N)

    # Simulation
    n_spikes = 0
    for i in range(1,N):

        # courant inject�
        if (t[i]>=t_on) and (t[i]<t_off):
            i_inj[i]  = i_amp[k]

        # potentiel membranaire
        v[i] = v[i-1] + dt/tau_m * ( -v[i-1] + R_m * i_inj[i-1]  )

        # detection de PA
        if v[i-1] > theta:
             v[i] = v_P

        # reset � 0
        if v[i-1] == v_P:
            v[i] = 0
            n_spikes = n_spikes + 1    # comptage de PA sortants
    
    # fr�quence de PA
    compte_pa[k] = n_spikes


# calculer la fr�quence th�orique et la repr�senter sur le m�me graphique
fI = np.zeros(len(i_amp))
for i in range(len(i_amp)):
    fI[i] = 1000/(tau_m*np.log(R_m*i_amp[i]/(R_m*i_amp[i]-theta)))

# Figure 
plt.figure(1); plt.show(); plt.clf()

# Courbe F-I
plt.subplot(111) 
plt.plot(i_amp, compte_pa, 'ro', label='simulation')
plt.plot(i_amp, fI, 'b-', label='frequence')
plt.xlabel('Courant (microA)')
plt.ylabel('Frequence, Hz')
plt.legend(loc='upper left')
plt.draw()
