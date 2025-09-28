### k-Nearest Neighbor classifier

### compute_distances_two_loops

The complete code is as follows.

```python
def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    
    for i in range(num_test):
        for j in range(num_train):
            # L2 distant-formula: sqrt(sum((x_i - x_j)^2))
            diff = X[i] - self.X_train[j]    # difference in each dimension
            dists[i, j] = np.sqrt(np.sum(diff ** 2))
    
    return dists
```

#### explain

The main purpose of this function is calculate the `L2` distance between the test data and the training data.

+ input data:
  + `X`: test set , shape is `(num_test, D)`, representing `num_test` samples, each sample has D features.
  + `self.X_train`: train set, shape is `(num_train, D)`, representing `num_train` samples.
+ output data:
  + `dists`:  A `2D` matrix with shape `(num_test, num_train)`
    + `dists[i, j]` representing the distance between the `i-th` test sample and the `j-th` training sample.

#### example

Assumptions:

+ when the test set has two sample, each sample has two features:\

  ```lua
  [[1, 2],
   [4, 5]]
  ```

+ train set `self.X_train` has three sample:

  ```lua
  [[1, 0],
   [0, 1],
   [3, 4]]
  ```

The results of running dists are roughly as follows:

```lua
[[2.0, 1.41, 2.83],
 [5.0, 5.0, 1.41]]
```

The distances from the test sample `[1, 2]` to the three training points are `2.0, 1.41, 2.83`

The distances from the test sample `[4, 5]` to the three training points are `5.0, 5.0, 1.41`

### Inline Question 1

**The meaning of the distance matrix**

+ **row**: represents the distance between a test sample and all training samples.
+ **column**: represents the distance between a train sample and all test samples.
+ **brightness**: the greater the distance, the brighter; the smaller the distance, the darker.

**Why are there “apparently bright rows”?**

if a row brighter than others, is means that the distance between the test sample and all training samples is relatively large.

Possible reason include:

+ This test sample is very far from the training set as a whole, meaning its feature distribution differs from that of most training set samples.
+ In other words, it may be an area that is not well covered in the training set.

**Why has "apparently bright columns"**

If a columns that others, is means that the distance between the test sample and all training samples is relatively large.

Possible reason include:

+ The training sample is an outlier or noise sample that is significantly different from the majority of the test data.
+ It may be dirty data or a rare class example.

**So answer**

The bright row represent the distance between a test sample and all training samples is relatively large, this indicates that these test samples are special or isolated points in the data distribution.

The bright column represent the distance between a training sample and all test samples is relatively large, this indicates that these train samples may be outliers, noise data, or categories with very rare distributions.

