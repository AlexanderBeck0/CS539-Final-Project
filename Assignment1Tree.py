import numpy as np
import collections
# Note: please don't add any new package, you should solve this problem using only the packages above.
# However, importing the Python standard library is allowed: https://docs.python.org/3/library/
#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes) -- 60 points --
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `pytest -v test1.py` in the terminal.
'''
#TODO: adapt to work with our dataset
#-----------------------------------------------
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X: np.ndarray = X
        self.Y: np.ndarray = Y
        self.i: int = i or -1
        self.C: dict = C or {}
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        ID3 Tree is replaced with a stochastic version that selects the best attribute from a random subset of attributes.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        counts = collections.Counter(Y)
    
        # Get lenght
        yLength = len(Y)
    
        e = 0.0  # Initialize
    
        # Calculate entropy using the log_2 formula
        for yCount in counts.values():
            p = yCount / yLength
            if p > 0:  
                e -= p * np.log2(p)


        #########################################
        return e 
    
    
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        nodes = collections.defaultdict(list)

        # Use a hash table for different values of x -> y
        # Faster than original implementation since we aren't recomputing lists over and over again
        for x, y in zip(X, Y):
            nodes[x].append(y)
    
        ce = 0.0
        
        yLength = len(Y)
        # Iterate through each unique attribute value
        # for xVal, xCount in xCounts.items():
        for ys in nodes.values():
        
            # Calculate the probability of this attribute value
            p_x = len(ys) / yLength

            # Calculate the entropy of the subset of target labels
            entropyYGivenX = Tree.entropy(ys)
        
            # Sum the weighted entropies
            ce += p_x * entropyYGivenX
        
        return ce
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X) 
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        g = Tree.entropy(Y) - Tree.conditional_entropy(Y,X)


 
        #########################################
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X: np.ndarray, Y: np.ndarray):
        '''
            Find the best attribute to split the node. Changed from project 1 to a stochastic version. 
            
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #This remains the same as before, the search logic is all that changes.
        numFeatures = X.shape[0]
        bestGain = -np.inf
        i = 0
        
        #determine subset size as root of total features, at least 1
        subsetSize = max(1, int(np.sqrt(numFeatures)))

        #randomly select subset of features
        featureIndices = np.random.choice(numFeatures, size=subsetSize, replace=False)

        for c in featureIndices: #for each feature in the random subset
            currentFeature = X[c,:]
            
            newGain = Tree.information_gain(Y,currentFeature)
            
            if newGain > bestGain:
                #take new best
                bestGain = newGain
                i = c

            elif newGain == bestGain:
                #randomly choose between the two features
                if np.random.rand() > 0.5:
                    i = c

        #for c in range(numFeatures): OLD CODE
        #    currentFeature = X[c,:]
            
        #   newGain = Tree.information_gain(Y,currentFeature)
            
        #   if newGain > bestGain:
        #      bestGain = newGain
        #      i = c

        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        C = {}  #Initialize
   
        splitAttributeValues = np.unique(X[i,:])

        for value in splitAttributeValues:
            mask = (X[i,:] == value)
       
            xSubset = X[:, mask]
            ySubset = Y[mask]

            childNode = Node(X=xSubset, Y=ySubset)
       
            C[value] = childNode

        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        s = (np.unique(Y).size == 1)
        
        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attribute values. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    
        s = (np.unique(X, axis=1).shape[1] == 1)
 
        #########################################
        return s
    
            
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        y = collections.Counter(Y).most_common(1)[0][0]
 
        #########################################
        return y
    
    
    
    #--------------------------
    @staticmethod
    def build_tree(t: Node, max_depth: int, current_depth: int =0) -> None:
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape p by n.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #default prediction, will be overwritten if leaf.
        t.p = Tree.most_common(t.Y)
        
        if Tree.stop1(t.Y) or Tree.stop2(t.X) or max_depth == current_depth:
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
            return

        t.i = Tree.best_attribute(t.X, t.Y)
        
        # Split the data and get children nodes
        t.C = Tree.split(t.X, t.Y, t.i)
        
        if not t.C:
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
            return
        # Recursively build subtrees for each child
        for childNode in t.C.values():
            Tree.build_tree(childNode, max_depth=max_depth, current_depth=current_depth + 1)
 
        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X: np.typing.ArrayLike, Y: np.typing.ArrayLike, max_depth=np.inf) -> Node:
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t = Node(X, Y)
        Tree.build_tree(t, max_depth) # type: ignore
        #########################################
        return t
    
    
    
    #--------------------------
    @staticmethod
    def inference(t: Node, x: np.ndarray):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        if t.isleaf:
            y = t.p
            return y

        splitValue = x[t.i]
        if splitValue in t.C:
            return Tree.inference(t.C[splitValue], x)
        else:
            y = t.p
                
 
        #########################################
            return y
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        numInstances = X.shape[1]
        Y = np.zeros(numInstances, dtype=object)
        for i in range(numInstances):
            x = X[:, i]
            Y[i] = Tree.inference(t, x)


        #########################################
        return Y

    @staticmethod
    def inference_proba(t, x, classes):
        '''
            Given a decision tree and one data instance, infer the class probabilities recursively. 
            Returns a numpy array of probabilities (shape 2,) corresponding to the order in 'classes'.
        '''
        # --- Base Case: Leaf Node ---
        if t.isleaf:
            # Count the labels in this node
            counts = collections.Counter(t.Y)
            n = len(t.Y)
            
            # Create probability vector initialized to zeros
            proba = np.zeros(len(classes), dtype=float)
            
            for idx, cls in enumerate(classes):
                if n > 0:
                    # Calculate P(class) = Count(class) / Total
                    proba[idx] = counts[cls] / n
            return proba
        # --- Recursive Case: Internal Node ---
        splitValue = x[t.i]
        if splitValue in t.C:
            # Traverse to the child node
            return Tree.inference_proba(t.C[splitValue], x, classes)
        else:
            # Handling unseen value for this attribute: 
            # Default to the current node's probabilities (similar to a leaf prediction)
            counts = collections.Counter(t.Y)
            n = len(t.Y)
            proba = np.zeros(len(classes), dtype=float)
            for idx, cls in enumerate(classes):
                if n > 0:
                    proba[idx] = counts[cls] / n
            return proba

    @staticmethod
    def predict_proba(t, X):
        '''
            Given a decision tree and a dataset, predict the class probabilities on the dataset. 
            Returns a numpy matrix of shape (num_instances, num_classes).
        '''
        numInstances = X.shape[1]
        classes = sorted(np.unique(t.Y))
        numClasses = len(classes)
        
        # Initialize probability matrix
        proba_matrix = np.zeros((numInstances, numClasses), dtype=float)
        
        for i in range(numInstances):
            x = X[:, i]
            proba_matrix[i, :] = Tree.inference_proba(t, x, classes)
        
        return proba_matrix, classes
    #--------------------------
    @staticmethod
    def load_dataset(filename = 'data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        data = np.genfromtxt(filename, delimiter=',', dtype=str, skip_header=1)
        X = data[:, 1:]
        Y = data[:, 0]
        X=X.T
        #########################################
        return X,Y



