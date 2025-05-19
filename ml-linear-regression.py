# imports
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # ensure that all elements of X are represented as floating-point numbers
    X = X.astype(float)

    # compute the mean, maximum, and minimum values of the current feature
    mean_feature = np.mean(X, axis=0)
    max_feature = np.max(X, axis=0)
    min_feature = np.min(X, axis=0)
    # NumPy automatically broadcasts arrays of different shapes to match the larger array's shape, enabling element-wise operations between them.
    X = (X - mean_feature) / (max_feature - min_feature)

    # compute the mean, maximum, and minimum values of the label array y
    mean_lable = np.mean(y, axis=0)
    max_lable = np.max(y, axis=0)
    min_lable = np.min(y, axis=0)

    # Perform mean normalization on the lable.
    y = (y - mean_lable) / (max_lable - min_lable)   
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # number of rows in the existing array
    num_rows = X.shape[0]

    # create a column vector of 1's
    ones_column = np.ones((num_rows, 1))

    # concatenate the column of 1's before the existing array along the second axis
    X = np.column_stack((ones_column, X))    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    h_x = X.dot(theta) # Compute the hypothesis
    # Calculate the error between the predicted values and the actual values
    error  = np.subtract(h_x, y)
    squared_error= np.power(error, 2)
    sum = np.sum(squared_error)
    m = len(y)
    J=sum / (2*m)
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    m = len(X)
    for i in range(num_iters): 
        h_x = X.dot(theta) # Compute the hypothesis
        # Calculate the error between the predicted values and the actual values, transpose the result and multiply it by X
        error = X.T.dot(np.subtract(h_x, y))
        #update theta 
        theta = np.subtract(theta,((error / m) * alpha))
        #add the cost value of this iteration
        J_history.append(compute_cost(X,y,theta))

    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    pinvX=np.matmul(np.linalg.inv((np.matmul(X.T,X))),X.T)
    pinv_theta = np.matmul(pinvX,y)
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    m = len(X)
    for i in range(num_iters): 
        h_x = X.dot(theta) # Compute the hypothesis
        # Calculate the error between the predicted values and the actual values, transpose the result and multiply it by X
        error = np.dot(X.T , np.subtract(h_x, y))
        #update theta 
        theta = np.subtract(theta,((error / m) * alpha))
        J_history.append(compute_cost(X,y,theta))
        #add the cost value of this iteration
        if i > 0 and (J_history[i-1] - J_history[i]) < 0.00000001:
            break
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # Guess initial values for theta
    #theta=np.random.random(len(X_train[0]))
       # A loop that goes through all the alphas and puts the alpha with its cost value in a dictionary.
    for alpha in alphas:
        np.random.seed(42) #used to ensure reproducibility
        theta = np.random.random(X_train.shape[1])
        #train the model 
        theta , _ = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    while(len(selected_features) < 5):
        temp_features=[]
        np.random.seed(42)
        theta_rnd = np.random.random(len(selected_features) + 2)
        for feature_index, feature in enumerate(X_train[0]):
            if feature_index not in selected_features:
                # Add the feature to the selected set temporarily
                selected_features.append(feature_index)
                #These lines select all rows from the X_train and X_val matrices, but only the columns specified by selected_features.
                temp_X_train = apply_bias_trick(X_train[:, selected_features])
                temp_X_val = apply_bias_trick(X_val[:, selected_features])
                #train the model using the current set of selected features 
                theta, _ = efficient_gradient_descent(temp_X_train, y_train, theta_rnd, best_alpha, iterations)
                #evaluate its perfomance by calculating the cost or error on a validation set
                cost = compute_cost(temp_X_val,y_val,theta)                
                temp_features.append((feature_index,cost))
                # Remove the feature from the selected set
                selected_features.pop()
    
        #choose the feture that resulted in the best model
        # finds the tuple with the minimum cost based on the second element of each tuple and extracts the first 
        # element of the selected tuple, which is the index of the feature that resulted in the lowest cost
        best_features = min(temp_features, key=lambda x: x[1])
        best_feature = best_features[0]
        #add it to the selected set
        selected_features.append(best_feature)

    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    result_dict = {}

    # Iterate over all pairs of columns
    for i, col in enumerate(df_poly.columns):
        for new_col in df_poly.columns[i:]:

            # Construct new column name
            if col != new_col:
                feature_name = col + '*' + new_col
            else:
                feature_name = col + '^2'

            # Compute the new feature values
            new_col_values = df_poly[col] * df_poly[new_col]

            # Store the new feature values in the dictionary
            result_dict[feature_name] = new_col_values

    # Concatenate the original DataFrame with the dictionary of new features
    df_poly = pd.concat([df_poly, pd.DataFrame(result_dict)], axis=1)

    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly