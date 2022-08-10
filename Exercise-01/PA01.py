import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from sklearn.model_selection import KFold

#     First Empirical Distribution

gray_image = face(gray=True)
plt.figure(1)
plt.subplot(221)
plt.imshow(gray_image, cmap='gray')

gray_image_filtered = gaussian_filter(gray_image, sigma=0.1)
plt.subplot(222)
plt.imshow(gray_image_filtered, cmap='gray')

pdf = np.reshape(gray_image_filtered, (1, -1))
cdf = np.cumsum(pdf)
cdf = cdf / max(cdf)

print('Image shape is = ', gray_image_filtered.shape)
print('PDF shape is = ', pdf.shape)
print('CDF shape is = ', cdf.shape)
print()
print("*** Enter an integer as number of samples from distribution function.")

N = int(input("N="))
N_random = np.random.uniform(0, 1, N)
cdf_N_samples = np.searchsorted(cdf, N_random)

pdf[:] = 0
for i in range(N):
	pdf[0][cdf_N_samples[i]] = 1

blurry_image = np.reshape(pdf, (gray_image.shape[0], gray_image.shape[1]))
plt.subplot(223)
plt.imshow(blurry_image, cmap='gray')

#     Parzen Windows

win_size = 3
win = np.ones((win_size, win_size))
win = win / (np.shape(win)[0] * np.shape(win)[1])
Parzen_image = convolve2d(blurry_image, win)
plt.subplot(224)
plt.imshow(Parzen_image, cmap="gray")
plt.show()

#   Cross Validation  

window_sizes = [5, 15, 21, 9]
log_liklihood = []
for win_size in window_sizes:
    win = np.ones((win_size, win_size))
    win = win / (np.shape(win)[0] * np.shape(win)[1])
    cv = KFold(n_splits=5)
    sum_log_cdf = 0.0
    for train_index, test_index in cv.split(blurry_image):
        train_data, test_data = blurry_image[train_index], blurry_image[test_index]
        Parzen_train = convolve2d(train_data, win)
        pdf_train = np.reshape(Parzen_train, (1, -1))
        cdf_train = np.cumsum(pdf_train)
        cdf_train = cdf_train / max(cdf_train)
        sum_log_cdf = sum_log_cdf - np.log(np.sum(cdf_train))
    log_liklihood.append(sum_log_cdf)
    print()
    print('-Sum Log liklihood for window size', win_size, 'is :', sum_log_cdf)

best_kernel = np.where(log_liklihood == np.min(log_liklihood))
print()
print('The best window size is :', window_sizes[best_kernel[0][0]]) 
    
win_size = best_kernel[0][0]
win = np.ones((win_size, win_size))
win = win / (np.shape(win)[0] * np.shape(win)[1])
Parzen_image = convolve2d(blurry_image, win)
plt.figure(2)
plt.imshow(Parzen_image, cmap="gray")
plt.show()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    