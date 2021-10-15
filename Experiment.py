import matplotlib.pyplot as plt
import numpy
from DataGenerationRadar1D import GenerateData
from KalmanFilter import KalmanFilter

opt = {
        "initialDistance": 8,
        "stopTime": 1,
        "movementRange": 1,
        "frequency": 2,
        "SporadicError": 3
    }

timeAxis, distValues, velValues, truthDistValues, truthVelValues = GenerateData(type="Sinus", options=opt)

# test = numpy.array(distValues)
# numpy.savetxt("test.csv", test, delimiter=";")

plt.figure()
plt.plot(timeAxis, distValues)
plt.plot(timeAxis, velValues)
plt.plot(timeAxis, truthDistValues)
plt.plot(timeAxis, truthVelValues)
plt.xlabel("time in s")
plt.legend(["Distance", "Velocity", "Truth distance", "Truth velocity"])
plt.title("Measurement Data of a 1D Radar Sensor")
plt.grid(True)

'''
Aufgabe:
1. Implementieren Sie ein Kalman-Filter, das die Messdaten als Eingangsdaten nimmt.
2. Testen Sie das Kalman-Filter mit verschiedener Objektbewegungsarten.
'''

# Initialisierung
H = numpy.array([1, 1, 0])          # Verknüpfungsmatrix H
                                    # [1, 1, 0] da drei Werte (x, v und a), x und v vom Sensor als Messwert
# H = H.reshape(3,1)
deltat = timeAxis[1] - timeAxis[0]

Phi = numpy.array([                 # Übergangsmatrix Phi: Gleichungssystem:
    [1, 1 * deltat, 0.5 * deltat**2],                    # h = h + v * t + 1/2 a * t^2
    [0, 1, 1 * deltat],                      # v = v + a * t
    [0, 0, 1 * deltat]                       # a = a 
])
'''
Wird hier zwingend die Beschleunigung benötigt? -> Sensor übergibt direkt Geschwindigkeitswerte
'''
PFaktor = 10                    # großer Wert; hier ggf. auch Zufallswerte

# Hier Ihr Kalman-Filter initialisieren
sinit = numpy.array([distValues[0], velValues[0], 0])
sinit = sinit.reshape(3,1)
estest = numpy.array([0.05, 0.1, 0.1]).reshape(3, 1)# numpy.ones(3).reshape(3,1)
emtest = numpy.array([0.05, 0.06, 0.1]).reshape(3, 1)# numpy.ones(3).reshape(3,1)

kFilter = KalmanFilter(sinit, PFaktor, estest, emtest, 1, H, Phi)

results = []



for i in range(numpy.size(timeAxis)):
    # hier die Daten ins Kalman-Filter eingeben
    # output = kFilter.Step(input)
    input = numpy.array([distValues[i], velValues[i], 0])
    input = input.reshape(1,3)
    result, P = kFilter.Step(input)
    results.append(result)
    print(result)


# plt.plot(timeAxis, results)
plt.show()

# Hier das Ergebnis über die Zeit plotten.

# Um wie viel hat sich die Messgenauigkeit verbessert?
# Wie beeinflussen die Schätzung der Kovarianzmatrix Q und R die Genauigkeit
# Fügen Sie zufällige Messfehler mit der Parameter "SporadicError" hinzu, wie verhält sich das Kalman Filter?