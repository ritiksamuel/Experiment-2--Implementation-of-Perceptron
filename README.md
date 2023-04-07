# Ex no.2
# Date: 29.09.2022
# Implementation-of-Perceptron

## AIM:
To implement a perceptron for classification using Python

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
* A __Perceptron__ is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.
* A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.
* The important Key points to be focused to implement a perceptron:
* __Models have to be trained with a high number of already classified samples.__ It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
* Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, __scaling.__
* Perceptron requires __linearly separable samples__ to achieve convergence.
## The math of Perceptron
* If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
<br>![for2](./formula2.png)
* A threshold function, usually __Heaviside or sign functions__, maps the scalar value to a binary output:
<br>![for1](./formula1.png)
* Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.


## ALGORITHM:
1. Importing the libraries
2. Importing the dataset
3. Plot the data to verify the linear separable dataset and consider only two classes
4. Convert the data set to scale the data to uniform range by using Feature scaling
<br>![for2](./formula3.png)
5. Split the dataset for training and testing
6. Define the input vector ‘X’ from the training dataset
7. Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2
8. Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
9. Assign the learning rate
10. For ‘N ‘ iterations ,do the following:
<br>![for4](./formula4.png)
11. Plot the error for each iteration 
12. Print the accuracy

## PROGRAM:
### Libraries
```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
```

### Class Perceptron
```
class Perceptron:
  def __init__(self, learning_rate=0.1):
    self.learning_rate = learning_rate
    self._b = 0.0
    self._w = None
    self.misclassified_samples = []
  def fit(self, x: np.array, y: np.array, n_iter=10):
    self._b = 0.0
    self._w = np.zeros(x.shape[1])
    self.misclassified_samples = []
    for _ in range(n_iter):
      errors = 0
      for xi,yi in zip(x,y):
        update = self.learning_rate * (yi-self.predict(xi))
        self._b += update
        self._w += update*xi
        errors += int(update !=0)
      self.misclassified_samples.append(errors)
  def f(self,x:np.array) -> float:
    return np.dot(x,self._w) + self._b
  def predict(self, x:np.array):
    return np.where(self.f(x) >= 0,1,-1) 
```

```
df = pd.read_csv("C:\Users\chief\OneDrive\Documents\Experiment-2--Implementation-of-Perceptron\IRIS.csv")
df.head()

y = df.iloc[:,-1].values
y

x = df.iloc[:,:-1].values
x

x = x[0:100,0:2]
y = y[0:100]

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')

plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',
            label='Versicolour')

plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')

plt.show()

y = np.where(y=="Iris-setosa",1,-1)
y

x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,
                                                    random_state=0)

classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)


plt.plot(range(1, len(classifier.misclassified_samples) + 1),
         classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()

print("accuracy = " , accuracy_score(classifier.predict(x_test), y_test)*100)
```

## Output:

### Dataset
![df](./out/df.png)

### Scatterplot
![plot](./out/plot.png)

### Y- values
![val](./out/y.png)

### Error Plot
![val](./out/error.png)

### Accuracy
![acc](./out/accuracy.png)

# Result:
Thus a perceptron for classification is implemented using python
