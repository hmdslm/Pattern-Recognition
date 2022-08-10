import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestRegressor

gray_image = face(gray=True)
plt.figure(figsize=(20,10))
plt.subplot(231)
plt.imshow(gray_image, cmap='gray')

gray_image_filtered = gaussian_filter(gray_image, sigma=0.1)
plt.subplot(232)
plt.imshow(gray_image_filtered, cmap='gray')

pdf = np.reshape(gray_image_filtered, (1, -1))
cdf = np.cumsum(pdf)
cdf = cdf / max(cdf)

N = 100000
M = N
N_random = np.random.uniform(0, 1, N)
cdf_N_samples = np.searchsorted(cdf, N_random)

pdf[:] = 0
random_pdf = np.copy(pdf)

for i in range(N):
	pdf[0][cdf_N_samples[i]] = 1
    
random_numbers = np.random.randint(0, high=pdf.size, size=M)
for i in range(M):
	random_pdf[0][random_numbers[i]] = 1

blurry_image = np.reshape(pdf, (gray_image.shape[0], gray_image.shape[1]))
plt.subplot(233)
plt.imshow(blurry_image, cmap='gray')

random_image = np.reshape(random_pdf, (gray_image.shape[0], gray_image.shape[1]))
plt.subplot(234)
plt.imshow(random_image, cmap='gray')

training_data = np.ndarray(shape=(N+M, 2))
training_lables = np.ndarray(shape=(N+M))

for i in range(N):
    training_data[i][0] = int(cdf_N_samples[i]/gray_image.shape[1])
    training_data[i][1] = int(cdf_N_samples[i]%gray_image.shape[1])
    training_lables[i] = 1

for i in range(M):
    training_data[N+i][0] = int(random_numbers[i]/gray_image.shape[1])
    training_data[N+i][1] = int(random_numbers[i]%gray_image.shape[1])
    training_lables[N+i] = 0

model = RandomForestRegressor(n_estimators=10, max_depth=20)
model.fit(training_data, training_lables)

test_data = np.ndarray(shape=(pdf.size, 2))
for i in range(pdf.size):
    test_data[i][0] = int(i/gray_image.shape[1])
    test_data[i][1] = int(i%gray_image.shape[1])

predicted_lables = model.predict(test_data)
predicted_image = np.ndarray(shape=(gray_image.shape[0], gray_image.shape[1]))

for i in range(test_data.shape[0]):
    predicted_image[int(test_data[i][0])][int(test_data[i][1])] = predicted_lables[i]

plt.subplot(235)
plt.imshow(predicted_image, cmap='gray')

plt.show() 
    
    
    
    
    
    
    
    
    
    