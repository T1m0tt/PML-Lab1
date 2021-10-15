import numpy as np

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
        self.H = Hinit                          # Verknüpfungsmatrix
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
        # Vorhersage des Prädiktionsfehlers
        Pp = np.dot(np.dot(self.Phi, self.P), self.Phi.T) + self.Q

        # Kalman-Verstärkung
        K = np.divide(np.dot(Pp, self.H.T), (self.R + np.dot(np.dot(self.H, Pp), self.H.T)))

        # Aktualisierung der Kovarianzmatrix des Schätzfehlers
        self.P = np.dot((np.identity(self.H.size) - np.dot(K, self.H)), Pp)
        
        # Verbesserung der Schätzung
        epm = m - np.dot(self.H, sp)
        self.s = sp + np.dot(K, epm.T)
        
        return self.s[0], self.P
