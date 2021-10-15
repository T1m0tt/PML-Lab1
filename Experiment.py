import matplotlib.pyplot as plt
import numpy
from DataGenerationRadar1D import GenerateData
from KalmanFilter import KalmanFilter

optSinus = {
        "initialDistance": 8,
        "stopTime": 1,
        "movementRange": 1,
        "frequency": 2,
        "SporadicError": 3
    }

optConstantAcceleration = {
        "initialDistance": 8,
        "stopTime": 1,
        "initialVelocity": 3,
        "acceleration": 10,
        "movementRange": 1,
        "frequency": 2,
        "SporadicError": 0
    }

timeAxis, distValues, velValues, truthDistValues, truthVelValues = GenerateData(type="Sinus", options=optSinus)
timeAxis, distValues, velValues, truthDistValues, truthVelValues = GenerateData(type="ConstantAcceleration", options=optConstantAcceleration)

# test = numpy.array(truthDistValues)
# numpy.savetxt("truthTest.csv", test, delimiter=";")

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
H = numpy.array([
    [1, 0, 0],
    [0, 1, 0]]
    )          
# H = H.reshape(3,1)
deltat = timeAxis[1] - timeAxis[0]

Phi = numpy.array([                 # Übergangsmatrix Phi: Gleichungssystem:
    [1, 1 * deltat, 0.5 * deltat**2],                    # h = h + v * t + 1/2 a * t^2
    [0, 1, 1 * deltat],                      # v = v + a * t
    [0, 0, 1]                       # a = a 
])

PFaktor = 1e1                    # großer Wert; hier ggf. auch Zufallswerte

sinit = numpy.array([distValues[0], velValues[0], 0])

estest = numpy.array([0.2, 0.5, 0.1]).reshape(3, 1)
emtest = numpy.array([0.01, 0.0025]).reshape(2, 1)

# Hier Ihr Kalman-Filter initialisieren
kFilter = KalmanFilter(sinit, PFaktor, estest, emtest, 1, H, Phi)

resultsh = []
resultsv = []


for i in range(numpy.size(timeAxis)):
    # hier die Daten ins Kalman-Filter eingeben
    # output = kFilter.Step(input)
    input = numpy.array([distValues[i], velValues[i]]).reshape(2,1)
    resulth, resultv, P = kFilter.Step(input)
    resultsh.append(resulth)
    resultsv.append(resultv)
    # print(result)

plt.figure()
plt.plot(timeAxis, distValues)
plt.plot(timeAxis, resultsh, linestyle='dashed')
plt.plot(timeAxis, truthDistValues, linestyle='dotted')
plt.xlabel("time in s")
plt.legend(["Distance measured", "Distance filtered", "Truth distance"])
plt.title("Measurement Distance - Data of a 1D Radar Sensor with Kalman-Filter")
plt.grid(True)

plt.figure()
plt.plot(timeAxis, velValues)
plt.plot(timeAxis, resultsv, linestyle='dashed')
plt.plot(timeAxis, truthVelValues, linestyle='dotted')
plt.xlabel("time in s")
plt.legend(["Velocity measured", "Velocity filtered", "Truth velocity"])
plt.title("Measurement Velocity - Data of a 1D Radar Sensor with Kalman-Filter")
plt.grid(True)
plt.show()

# Hier das Ergebnis über die Zeit plotten.

# Um wie viel hat sich die Messgenauigkeit verbessert?
# Wie beeinflussen die Schätzung der Kovarianzmatrix Q und R die Genauigkeit
# Fügen Sie zufällige Messfehler mit der Parameter "SporadicError" hinzu, wie verhält sich das Kalman Filter?