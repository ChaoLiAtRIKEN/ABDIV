import numpy as np
import jax
import jax.numpy as jnp
# This is the first piece of codes for trying basic functions and frameworks.

# 0. Basic functions
def normalize_to_one(u):
    """
    Normalize a matrix so that the sum of its absolute values equals one.
    """
    return np.abs(u) / np.sum(np.abs(u))

@jax.jit
def log_tsallis(x, alpha):
    """
    The Tsallis Logrithm function required in the formula
    """
    if alpha != 0:
        return (x ** alpha - 1)/alpha
    else:
        return np.log(x)
    
def exp_tsallis(x, alpha):
    """
    The Tsallis exponent function, which is the inverse of Tsallis logirhtm function
    """
    if alpha != 0:
        return np.where(1+alpha*x<=0, 0 ** (1/alpha), (1+alpha*x) ** (1/alpha))
    else:
        return np.exp(x)
    

# 1. Data generalziation
size_data = (5, 2)
rank_factors = 2
U = [normalize_to_one(np.random.normal(0,1,size=[i, rank_factors])) for i in size_data]

data = np.matmul(U[0], U[1].T)

# 2. Decomposition algorithm
def model(theta):
    U, V = theta
    return jnp.matmul(U, V.T)

def loss_fn(theta, x):
    estimation_data = model(theta)
    return jnp.sum(jnp.square(x - estimation_data))

def grad_loss(theta, x):
    """
    Return the gradient of the loss function, wrt. the two factors
    """
    grad_U, grad_V = jax.grad(loss_fn)(theta, x)
    return grad_U, grad_V


def update(theta, x, ab, lr=0.1):
    alpha, beta = ab
    grad_U, grad_V = grad_loss(theta, x)
    U, V = theta
    U_new = U * exp_tsallis((-lr/U ** (alpha+beta-1)) * grad_U, beta)
    V_new = V * exp_tsallis((-lr/V ** (alpha+beta-1)) * grad_V, beta)

    return U_new/jnp.sum(U_new), V_new/jnp.sum(V_new)



# 3. Evaluation
ab = (1.0, 0.0)
theta = [normalize_to_one(np.random.normal(0,1,size=[i, rank_factors])) for i in size_data]

for _ in range(1000):
    theta = update(theta, data, ab, lr=0.3)
    loss = loss_fn(theta, data)
    print('Loss: ', loss)

print(U[0])

# print(exp_tsallis(log_tsallis(U[0], 2), 2))
