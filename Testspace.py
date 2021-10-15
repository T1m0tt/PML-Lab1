import numpy as np
from numpy.core.numerictypes import sctype2char

s1 = 7.989213093565174
s2 = 12.562317597191647
s3 = 0

s = np.array([s1, s2, s3]).reshape(3,1)

m = s

deltat = 0.01

Phi = np.array([                 # Übergangsmatrix Phi: Gleichungssystem:
    [1, 1 * deltat, 0.5 * deltat**2],                    # h = h + v * t + 1/2 a * t^2
    [0, 1, 1 * deltat],                      # v = v + a * t
    [0, 0, 1 * deltat]                       # a = a 
])

P = np.identity(3) * 1000

H = np.array([1, 0, 0])

es = np.array([0.005, 0.001, 0.001]).reshape(3, 1)
em = np.array([0.005, 0.006, 0.001]).reshape(3, 1)
Q = np.dot(es, es.T)    
R = np.dot(em, em.T) 



# Vorhersage des Zustands
sp = np.dot(Phi, s)
print(sp)
# Vorhersage des Prädiktionsfehlers
Pp = np.dot(np.dot(Phi, P), Phi.T) + Q
print(Pp)
# Kalman-Verstärkung
K = np.divide(np.dot(Pp, H.T), (R + np.dot(np.dot(H, Pp), H.T)))
# Aktualisierung der Kovarianzmatrix des Schätzfehlers
P = np.dot((np.identity(H.size) - np.dot(K, H)), Pp)

# Verbesserung der Schätzung
epm = m - np.dot(H, sp)
print(np.dot(K, epm))
s = sp + np.dot(K, epm)

print(s[0])
print()