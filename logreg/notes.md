In linear regression, we model our data with the linear function

$$
f(w,b) = wx + b
$$

Where $w$ is a vector containing weights, and $b$ is a vector containing the bias.

In logistic regression, we don't want the output to be continuous, but we want **probability**.

To model probability, we apply the **sigmoid function** to the linear model. The sigmoid function is defined as

$$
S(x) = \frac{1}{1 + e^{-x}}
$$

where we apply our linear model to $x$. Thus

$$
\hat{y} = h_\theta (x) = \frac{1}{1 + e^{-wx+b}}
$$

where $\hat{y}$ and $h_\theta (x)$ represent predicted probabilities generated using the parameters $w$ and $b$.

The sigmoid function outputs a probability between 0 and 1.

The cost function used is not MSE like for linear regression, but the **cross entropy**.

Cross entropy optimizes $w$ and $b$.

Update rules are defined as

$$
w_{new} = w_{old} - \alpha \cdot dw
$$
$$
b_{new} = b_{old} - \alpha \cdot db
$$

where $\alpha$ is the learning rate and $dw$ and $db$ are the derivatives of the weights and bias.

The formulas for $dw$ and $db$ are the same as they are for linear regression.