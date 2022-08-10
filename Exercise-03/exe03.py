import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

N_clusters = 5            # Number of Gaussian Clusters
N_B = 20                  # Number of Uniform Clusters
N_points = 500            # Number of points in Clusters
K_test = 3*N_clusters     # Number of k to test data

cov = [[2, 0], [0, 2]]
Gauss_data = np.empty((0,2))
Gauss_lables = np.empty((0, 0))
for i in range (N_clusters):
    mean = np.array([np.random.randint(50), np.random.randint(50)])
    Gauss_data = np.append(Gauss_data, np.random.multivariate_normal(mean, cov, N_points), axis=0)
    Gauss_lables = np.append(Gauss_lables, np.full((1, N_points), i))
plt.figure(figsize=(15,13))
plt.scatter(Gauss_data[:,0], Gauss_data[:,1], c=Gauss_lables)
plt.show()
plt.close()

Uniform_data = np.empty((0,N_points,2))
for i in range (N_B):
    Uniform = np.random.uniform(0, 50, (N_points, 2))
    Uniform_data = np.append(Uniform_data, [Uniform], axis=0)

Gauss_Wk = np.empty((0,2))
Uniform_Wks = np.empty((0,N_B))
Uniform_WkMean = np.empty((0,2))
STDV_k = np.empty((0,0))
Gap = np.empty((0,2))
for i in range(K_test):
    Gauss_kmeans = KMeans(n_clusters=i+1, random_state=0).fit(Gauss_data)
    Gauss_lables = Gauss_kmeans.labels_
    Gauss_centers = Gauss_kmeans.cluster_centers_
    Gauss_inertia = Gauss_kmeans.inertia_
    Gauss_Wk = np.append(Gauss_Wk, [[Gauss_inertia, i+1]], axis=0)
    plt.figure(figsize=(15,13))
    plt.scatter(Gauss_data[:,0], Gauss_data[:,1], c=Gauss_lables)
    plt.scatter(Gauss_centers[:,0], Gauss_centers[:,1], s=200, c='red')
    plt.show()
    plt.close()
    Uniform_Wk = np.empty((0,0))
    
    for j in range(N_B):
        Uniform_kmeans = KMeans(n_clusters=i+1, random_state=0).fit(Uniform_data[j])
        Uniform_lables = Uniform_kmeans.labels_
        Uniform_centers = Uniform_kmeans.cluster_centers_
        Uniform_inertia = Uniform_kmeans.inertia_
        Uniform_Wk = np.append(Uniform_Wk, [Uniform_inertia])
    plt.figure(figsize=(15,13))
    plt.scatter(Uniform_data[j][:,0], Uniform_data[j][:,1], c=Uniform_lables)
    plt.scatter(Uniform_centers[:,0], Uniform_centers[:,1], s=200, c='red')
    plt.show()
    plt.close()
    Uniform_Wks = np.append(Uniform_Wks, [Uniform_Wk], axis=0)
    Uniform_Wk = np.log(Uniform_Wk)
    Uniform_WkMean = np.append(Uniform_WkMean, [[np.average(Uniform_Wk), i+1]], axis=0)
    STDV_k = np.append(STDV_k, [np.std(Uniform_Wk)])
    Gap = np.append(Gap, [[Uniform_WkMean[i,0]-np.log(Gauss_Wk[i,0]), i+1]], axis=0)

STDV_k = STDV_k * np.sqrt(1.0 + 1.0 / N_B)
plt.figure(figsize=(15,13))
plt.plot(Uniform_WkMean[:,1], Uniform_WkMean[:,0], c='blue')
plt.scatter(Uniform_WkMean[:,1], Uniform_WkMean[:,0], c='blue')
plt.plot(Gauss_Wk[:,1], np.log(Gauss_Wk[:,0]), c='red')
plt.scatter(Gauss_Wk[:,1], np.log(Gauss_Wk[:,0]), c='red')
plt.show()
plt.close()    
   
plt.figure(figsize=(15,13))
plt.errorbar(Gap[:,1], Gap[:,0], yerr=STDV_k, uplims=True, lolims=True, label='Gap')
plt.scatter(Gap[:,1], Gap[:,0], c='red')
plt.show()
plt.close()   
   
for i in range(K_test):
    if i+1 == K_test:
        print('Try with a larger K_test')
        break
    if Gap[i,0] >= (Gap[i+1,0] - STDV_k[i+1]):
        print('********************')
        print('The best K is :', i+1)
        print('********************')
        break
        
    
    
    
    
    
    
    