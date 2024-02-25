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

def log_tsallis(x, alpha):
    if alpha != 0:
        return (x ** alpha - 1)/alpha
    else:
        return np.log(x)
    
def exp_tsallis(x, alpha):
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

def grad_loss(theta, x, ab):
    grad_U, grad_V = jax.grad(loss_fn)(theta, x)


def update(theta, x, ab, lr=0.1):
    pass



# 3. Evaluation
# print(np.sum(U[1]))

print(exp_tsallis(log_tsallis(3,0),0))