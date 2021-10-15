class KalmanFilter:
    
    # Initialisierung von Kalman Filter
    def __init__(self, sinit, Pinit, esinit, eminit, deltat, Hinit, Phiinit):
        self.s = sinit                          # s[-1]; Schätzwert
        self.P = np.identity(Hinit.size) * Pinit# Einheitsmatrix mit großer Konstante initialisiert; P[-1]
        self.es = esinit                        # Modellfehler
        self.em = eminit                        # Messfehler
        self.deltat = deltat                    # Zeitschritt t
        self.Q = np.dot(self.es, self.es.T)     # Prozessrauschen
        self.R = np.dot(self.em, self.em.T)     # Kovarianzmatrix des Prozessrauschens
        self.H = Hinit.reshape(1,3)             # Verknüpfungsmatrix
        self.Phi = Phiinit                      # Übergangsmatrix

    # Diese Funktion nimmt die Messwerten und gibt 
    # das Ergebnis des Kalman Filters zurück
    #
    # Bitte hier geeignete Eingabe- und Rückgabeparametern ergänzen
    def Step(self, m):
        '''
        m: Messwert des aktuellen Schritts -> m[k]
        '''
        # Vorhersage des Zustands
        sp = np.dot(self.Phi, self.s)
        # print(sp)
        # Vorhersage des Prädiktionsfehlers
        Pp = np.dot(np.dot(self.Phi, self.P), self.Phi.T) + self.Q
        # print(Pp)
        # Kalman-Verstärkung
        # K = np.divide(np.dot(Pp, self.H.T), (self.R + np.dot(np.dot(self.H, Pp), self.H.T)))
        # K = np.dot(np.dot(Pp, self.H.T),(self.R + np.dot(np.dot(self.H, Pp), self.H.T)))

        K_z = np.dot(Pp, self.H.T)
        K_n = np.linalg.inv(self.R + np.dot(np.dot(self.H, Pp), self.H.T))
        K = np.dot(K_n,K_z)
        # Aktualisierung der Kovarianzmatrix des Schätzfehlers
        self.P = np.dot((np.identity(self.H.size) - np.dot(K, self.H)), Pp)
        
        # Verbesserung der Schätzung
        x = np.dot(self.H, sp)
        epm = m - x
        # print(np.dot(K, epm.T))
        self.s = sp + np.dot(K, epm.T)

        print(self.s[0])
        
        return self.s[0], self.P


#############################################################################

# Initialisierung
H = numpy.array([1, 0, 0])          # Verknüpfungsmatrix H
                                    # [1, 1, 0] da drei Werte (x, v und a), x und v vom Sensor als Messwert
# H = H.reshape(3,1)
deltat = timeAxis[1] - timeAxis[0]

Phi = numpy.array([                 # Übergangsmatrix Phi: Gleichungssystem:
    [1, 1 * deltat, 0.5 * deltat**2],                    # h = h + v * t + 1/2 a * t^2
    [0, 1, 1 * deltat],                      # v = v + a * t
    [0, 0, 1 * deltat]                       # a = a 
])

sinit = numpy.array([distValues[0], velValues[0], 0])
sinit = sinit.reshape(3,1)
estest = numpy.array([0.05, 0.1, 0.1]).reshape(3, 1)# numpy.ones(3).reshape(3,1)
emtest = numpy.array([0.05, 0.06, 0.1]).reshape(3, 1)# numpy.ones(3).reshape(3,1)

PFaktor = 1e2                    # großer Wert; hier ggf. auch Zufallswerte


    input = numpy.array([distValues[i], velValues[i], 0])
    input = input.reshape(1,3)