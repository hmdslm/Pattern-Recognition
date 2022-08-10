import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor

moons_size = 10000
N = 200
M = N
data, lables = make_moons(n_samples=moons_size, noise=0.05)

plt.figure(figsize=(15,10))
plt.scatter(data[:,0], data[:,1], c=lables)

random_numbers = np.random.randint(0, high=moons_size, size=N)
training_data = np.ndarray(shape=(N+M, 2))
training_lables = np.ndarray(shape=(N+M))

for i in range(N):
    training_data[i][0] = data[random_numbers[i]][0]
    training_data[i][1] = data[random_numbers[i]][1]
    training_lables[i] = 0
#    training_lables[i] = lables[random_numbers[i]]

x_random = 5 * np.random.random_sample(M) - 2
y_random = 3.5 * np.random.random_sample(M) - 1.5

for i in range(M):
    training_data[N+i][0] = x_random[i]
    training_data[N+i][1] = y_random[i]
    training_lables[N+i] = 2

plt.figure(figsize=(15,10))
plt.scatter(training_data[:,0], training_data[:,1], c=training_lables)
    
model = ExtraTreesRegressor(n_estimators=10, max_depth=20)
#model = ExtraTreeRegressor(max_depth=10)
model.fit(training_data, training_lables)

predicted_lables = model.predict(data) 

plt.figure(figsize=(15,10))
plt.scatter(data[:,0], data[:,1], c=predicted_lables)


