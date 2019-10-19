# Evaluating a Machine Learning Algorithm
> #### A TOOL BOX - ‘WHAT TO TRY NEXT?’

## Overview
With abundance of easy-to-use ML Libraries available, it is often appealing to apply them and achieve greater than 80% prediction accuracy in most cases. But, **‘WHAT TO TRY NEXT?’** is a question that buzz me, and may be other aspiring Data Scientists, a lot.
<br />
<br />
During my course **‘Machine Learning – Stanford Online’** at Coursera, **Prof. Andrew Ng** helped me sail through it. I hope this article, which briefs his explanation during one of his lectures, will help many of us to understand the importance of ‘debugging or diagnosing a learning algorithm’.
<br />
<br />

To start with, let’s call out all the possibilities or **‘WHAT TO TRY NEXT?’** when a hypothesis makes unacceptably large errors in its predictions or when there is a need to improve our hypothesis:


| No. | <p align="left">          **‘WHAT TO TRY NEXT?’**               </p>|
|---  |---------------------------------------------------------------------|
| 1.  | <p align="left"> Try Smaller Set of Features                    </p>|
| 2.  | <p align="left"> Add New Features                               </p>|
| 3.  | <p align="left"> Add Polynomial Features                        </p>|
| 4.  | <p align="left"> Decrease Regularization Parameter ($\lambda$)  </p>|
| 5.  | <p align="left"> Increase Regularization Parameter ($\lambda$)  </p>|
| 6.  | <p align="left"> Get More Training Examples                     </p>|



_<div style="text-align: right"> We will revisit this table to make smart choices and create our **TOOL BOX**. </div>_

The above-mentioned diagnosis will basically help to find a **Bias Variance Trade Off**.  
Let’s visualize this concept briefly with a simple figure to illustrate the _overfitting_ (High Variance) and _underfitting_ (High Bias). 

## The Bias Variance Trade Off

_Fundamentally, the question of "the best model" is about finding a sweet spot in the tradeoff between bias and variance._
<br />
Here is a link to [The Bias Variance Trade Off](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html#The-Bias-variance-trade-off) explained beautifully by Jake VanderPlas in Python Data Science Handbook.

Following code visualizes over different _degrees of polynomial_.  
Please note that _overfitting_ and _underfitting_ can occur over different _regularization parameter_ and _training set size_.
> Source Code: [Scipy Lecture Notes](https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_bias_variance.html#bias-and-variance-of-polynomial-fit)

```
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def generating_func(x, err=0.5):
    return np.random.normal(10 - 1. / (x + 0.1), err)

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

n_samples = 8

np.random.seed(0)
x = 10 ** np.linspace(-2, 0, n_samples)
y = generating_func(x)

x_test = np.linspace(-0.2, 1.2, 1000)

titles = ['d = 1 (under-fit; high bias)',
          'd = 2',
          'd = 6 (over-fit; high variance)']
degrees = [1, 2, 6]

fig = plt.figure(figsize=(9, 3.5))
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.85, wspace=0.05)

for i, d in enumerate(degrees):
    ax = fig.add_subplot(131 + i, xticks=[], yticks=[])
    ax.scatter(x, y, marker='x', c='k', s=50)

    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(x[:, np.newaxis], y)
    ax.plot(x_test, model.predict(x_test[:, np.newaxis]), '-b')

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, 12)
    ax.set_xlabel('house size')
    if i == 0:
        ax.set_ylabel('price')

    ax.set_title(titles[i])
```

