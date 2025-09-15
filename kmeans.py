'''kmeans.py
Performs K-Means clustering
Tenzin Choden Thinley
CS 251/2: Data Analysis and Visualization
Fall 2024
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            # data: ndarray. shape=(num_samps, num_features)
            self.data = data.copy()
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        pass
        self.data = data
        self.num_features = self.data.shape[2]
        self.num_samps = len(self.data)

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        pass
        
        data_copy = self.data.copy()
        return data_copy
        

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)
        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        pass
        
        subtract = pt_1 - pt_2
        squared_sum= np.sum(subtract**2)
        dist = np.sqrt(squared_sum)
        return dist


    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        ''' 
        pass
        pt = pt.reshape(1 , -1)
        subtract = centroids - pt 
        squared_sum = np.sum(subtract**2 , axis = 1)
        dist = np.sqrt(squared_sum)

        return dist

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        pass

        rand_indices = np.random.choice(self.data.shape[0] , size = k , replace=False)
        rand_centroids = self.data[rand_indices]
        self.centroids = rand_centroids
        return rand_centroids
    
    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, p=2):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        if self.num_samps < k:
            raise RuntimeError('Cannot compute kmeans with #data samples < k!')
        if k < 1:
            raise RuntimeError('Cannot compute kmeans with k < 1!')
        
        # Initialize K-means variables
        self.k = k
        self.centroids = self.initialize(k)
        self.data_centroid_labels = self.update_labels(self.centroids)
        self.inertia = self.compute_inertia()
        # self.num_samps = self.data.shape[0]
        # self.num_features = self.data.shape[1]
        no_iter = 0

        # Do K-means as long as the max number of iterations is not met AND the absolute value of the difference between the previous and current centroid values is > `tol`
        

        while no_iter < max_iter: 
            new_centroids, centroid_diff = self.update_centroids(self.k, self.data_centroid_labels , self.centroids)
            new_data_labels = self.update_labels(new_centroids)
            # diff = np.abs(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.data_centroid_labels = new_data_labels
    
            
           
            if np.max(np.abs(centroid_diff)) < tol: 
                self.inertia = self.compute_inertia()
                # print(no_iter)
                return self.inertia, no_iter

            no_iter += 1

        self.inertia = self.compute_inertia()
        return self.inertia , no_iter

            


    def cluster_batch(self, k=2, n_iter=1, verbose=False, p=2):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        pass
        min_inert = np.inf
        min_centroids = None
        min_data_centroid_labels = None
        for i in range(n_iter): 
            inert, _ = self.cluster(k)
            if inert < min_inert: 
                min_inert = inert
                min_centroids = self.centroids.copy()
                min_data_centroid_labels = self.data_centroid_labels.copy()
        
        self.centroids = min_centroids
        self.data_centroid_labels = min_data_centroid_labels
        self.inertia = min_inert




    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        pass
        # Loop over the points, assigning them to the cluster with the closest centroid (this updates data cluster labels
        labels = np.zeros(self.num_samps, dtype=int)
        for i in range(self.num_samps): 
            distances = self.dist_pt_to_centroids(self.data[i], centroids)
            labels[i] = np.argmin(distances)
  
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        pass
        
        new_centroids = np.zeros((k, self.num_features))
        for i in range(k): 
            kth_cluster = self.data[data_centroid_labels == i]
            if len(kth_cluster) == 0: 
                rand_idx = np.random.choice(self.data.shape[0], size = 1)
                rand_centroid = self.data[rand_idx]
                new_centroids[i] = rand_centroid
            else: 
                kth_cluster_mean = np.mean(kth_cluster , axis = 0)
                new_centroids[i] = kth_cluster_mean

        centroid_diff = new_centroids - prev_centroids

        return new_centroids , centroid_diff

        

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        pass  
        # inertia_list = np.zeros(self.k)
        # for i in self.k:
        #     # kth_cluster_pt = 
        #     # pt_cent = self.dist_pt_to_centroids(kth_cluster_pt , self.centroids[i  , :])
        #     inertia_list[i] = np.mean(self.dist_pt_to_centroids((self.data[self.data_centroid_labels == i]), self.centroids[i , :]))
        # inertia = np.mean(inertia_list)
        # return inertia

        distance_list = np.zeros((self.data.shape[0], self.data.shape[1]))
        for i in range(self.data.shape[0]): 
            pt = self.data[i]
            # pt = pt.reshape(1 , -1)
            centroid = self.centroids[self.data_centroid_labels[i]]
            distance = self.dist_pt_to_centroids(pt, centroid)
            distance_list[i] = distance**2
        inertia = np.mean(distance_list)

        return inertia



    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        '''
        pass
        # 10 colors
        colors = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]   
        cmap = ListedColormap(colors)  

        plt.figure(figsize=(12, 8))
        plt.scatter(self.data[:, 0], self.data[: , 1] , c=self.data_centroid_labels, cmap=cmap)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', marker='*', label='Centroids' )
        plt.title("K-means Clustering")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
        
        

    # def elbow_plot(self, max_k):
    #     '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

    #     Parameters:
    #     -----------
    #     max_k: int. Run k-means with k=1,2,...,max_k.

    #     TODO:
    #     - Run k-means with k=1,2,...,max_k, record the inertia.
    #     - Make the plot with appropriate x label, and y label, x tick marks.
    #     '''
    #     pass
    #     inertia_list = np.zeros(max_k)
    #     for i in range(1, max_k+1): 
    #         inertia, diff = self.cluster(i)
    #         inertia_list[i-1] = inertia

    #     k_val = np.arange(1 , max_k+1)
    #     plt.plot(k_val , inertia_list)
    #     plt.xticks(np.arange(1, max_k + 1, 1))
    #     plt.title("Elbow Plot of Varying Number of Centroids")
    #     plt.xlabel("k clusters")
    #     plt.ylabel("Inertia")
        
    #     plt.show()

    def elbow_plot(self, max_k, n_iter=1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        pass
        inertia_list = np.zeros(max_k)

        for i in range(1, max_k+1): 
            self.cluster_batch(k=i, n_iter=n_iter)
            inertia_list[i-1] = self.inertia

        k_val = np.arange(1 , max_k+1)
        plt.plot(k_val , inertia_list)
        plt.xticks(np.arange(1, max_k + 1, 1))
        plt.title("Elbow Plot of Varying Number of Centroids")
        plt.xlabel("k clusters")
        plt.ylabel("Inertia")
        
        plt.show()


        
        

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        pass
        for i in range(self.data.shape[0]):
            pt = self.data[i]
            dist = self.dist_pt_to_centroids(pt , self.centroids)
            min_dist = dist.argmin()
            self.data[i] = self.centroids[min_dist]
        

        
