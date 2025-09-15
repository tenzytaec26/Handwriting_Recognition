'''rbf_net.py
Radial Basis Function Neural Network
Tenzin Choden Thinley
CS 251: Data Analysis and Visualization
Fall 2024
'''
import numpy as np
import kmeans
import scipy.linalg
import matplotlib.pyplot as plt


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None
        # k: number of hidden units as an instance variable called `k` (as in k clusters)
        self.k = num_hidden_units
        # number of classes (number of output units in network) as an instance variable
        self.num_classes = num_classes


    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        pass
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        pass
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        pass
        dist = np.zeros(len(data))
        avg_dist = np.zeros(self.k)
        for i in range(len(data)): 
            samp_centroid = centroids[cluster_assignments[i]]
            dist[i] = kmeans_obj.dist_pt_to_pt(data[i], samp_centroid )
        # print(dist)
        for i in range(len(centroids)): 
            avg_dist[i] = np.mean(dist[cluster_assignments == i])
        # print(avg_dist)
        return avg_dist

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        pass
        kmeans_obj = kmeans.KMeans(data)
        kmeans_obj.cluster_batch(k = self.k, n_iter=5 , verbose=False, p=2)
        self.prototypes = kmeans_obj.get_centroids()
        cluster_assignments = kmeans_obj.get_data_centroid_labels()
        self.wts = np.zeros((self.k + 1, self.num_classes))

        self.sigmas = self.avg_cluster_dist(data , self.prototypes, cluster_assignments , kmeans_obj)
        
        

    def linear_regression(self, A, y):
        '''Performs linear regression. Adapt your SciPy lstsq code from the linear regression project.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept column.
        '''
        pass
        # This is a helper function that will be used by the weight-finding code. Set it up just like you did in the linear regression project. Augment the input matrix (A) with a column of 1’s, and then all scipy.linalg.lstsq


        Ahat = np.hstack([np.ones([A.shape[0], 1]), A])
    
        c , _ , _ , _ = scipy.linalg.lstsq(Ahat , y)


        # intercept = c[0 , 0]
        # slope = c[1:].reshape(-1 , 1)

        # residuals = y - (Ahat @ c)
        # m_sse = np.mean(residuals ** 2)

        # total = np.sum((y - np.mean(y)) ** 2)
        # r2 = 1 - np.sum(residuals**2) / total
        # print(c.shape)

        return c
        
        

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        pass
        # Computes hidden layer activation values: Determines the similarity between hidden layer prototypes with the input data.
        hidden_layer = np.zeros((len(data), self.k))
        x = 1e-8
        # # print(data)
        for i in range(len(data)): 
            pt = data[i].reshape(1 , -1)

            # Finding distance manually
            subtract = self.prototypes - pt
            squared_sum = np.sum(subtract**2, axis =1)
            dist = np.sqrt(squared_sum)

            # Finding distance using numpy for comparison
            # dist = np.linalg.norm(self.prototypes - pt, axis=1)

            # standard deviation is sigmas.
            hidden_layer[i] = np.exp(-((dist**2)/(2*(self.sigmas**2)-x)))

        return hidden_layer


        
    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        pass
        # This is the function that computes Z, the activation values for the output layer. It takes H as input, augments it with a column of ones (H1), and multiplies it by the wts 
        # Z = H1 * wts
        # hidden_acts = np.atleast_2d(hidden_acts)

        ones = np.ones((len(hidden_acts), 1))
        hidden_acts = np.hstack((ones, hidden_acts))

        output = hidden_acts @ self.wts

        return output

        

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using linear regression. The regression is between the hidden layer activation (to the data) and the correct classes of each training sample. To solve for the weights going FROM all of the hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        pass
        # train: Train the network, given training features and classes. Use the helper functions:
            # – Call initialize to train the hidden layer (i.e. do the clustering and set the prototypes and sigmas)
            # – Call hidden act to compute H for the training data.
            # – For each output node
                # ∗ Construct the ideal column of Z for that class
                # ∗ Call linear regression to find the corresponding column of weights
        self.initialize(data)
        hidden_layer = self.hidden_act(data)
        # print(hidden_layer)
        # output = self.output_act(hidden_layer)
        one_hot_encoded_y = np.zeros((len(data), self.num_classes))
        weights = np.zeros((self.k+1, self.num_classes))
        # one hot encode:
        for i in range(len(data)):
            samp_class = y[i] 
            one_hot_encoded_y[i, samp_class] = 1

        for i in range(self.num_classes): 
            z = one_hot_encoded_y[: , i]
            weights[: , i] = self.linear_regression(hidden_layer, z)
        
            print(i, np.sum(z))
        
        self.wts = weights
        # print(weights)


    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        pass
        # predict: Predict the class of the input data.
            # – Compute H using hidden act
            # – Compute Z using output act
            # – Predict the class from Z using argmax

        hidden_act = self.hidden_act(data)
        z = self.output_act(hidden_act)
       
        predicted_class = np.argmax(z , axis=1)
        
        return predicted_class


    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        pass
        matched = np.sum(y == y_pred)
        accuracy = matched/(len(y))
        
        return accuracy

    def confusion_matrix(self, y, y_pred, num_classes):
        '''Creates a confusion matrix comparing the ground-truth and predicted class labels.

        Parameters:
        -----------
        y: The original labels 
        y_pred: Predicted class labels.
        num_classes: Number of classes in the dataset.

        Returns:
        -----------
        Confusion matrix.
        '''
        # Initialize the confusion matrix
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        labels = np.hstack((y.reshape(-1, 1), y_pred.reshape(-1, 1)))

        # Fill the matrix
        for label in labels:
            conf_matrix[label[0], label[1]] += 1

        return conf_matrix



    def plot_predictions(self, data, y, y_pred, num_samples=25):
        '''Displays a grid of test images with their true and predicted labels.

        Parameters:
        -----------
        data: Data from the images
        y: The original labels for the images. 
        y_pred: Predicted labels for the test images.
        num_samples: Number of images to display in the grid (must be a perfect square).
        '''

        data = data.reshape(data.shape[0], 28, 28)
        # Creating a square grid for my subplots
        grid_size = int(num_samples ** 0.5)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(22, 22))
        axes = axes.flatten()

        for i in range(num_samples):
            img = data[i]
            cur_y = y[i]
            cur_y_pred = y_pred[i]
            if cur_y == cur_y_pred: 
                color = 'green' 
            else: 
                color = 'red'

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"y: {cur_y}  y_pred: {cur_y_pred}", color=color, fontsize=8)
            axes[i].axis('off')
        

        plt.tight_layout()
        plt.show()
        