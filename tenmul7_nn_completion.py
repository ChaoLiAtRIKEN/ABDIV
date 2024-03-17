import numpy as np
# np.random.seed(420)
import jax.numpy as jnp
import functools, itertools

from tenmul7_nn import NeuroTN

def generate_TT_adj_matrix(order, rank, dim_mode):
    adjm = np.diag(np.full((order-1,),rank), 1)
    adjm = adjm + adjm.transpose()
    np.fill_diagonal(adjm, dim_mode)

    return adjm

def generate_TR_adj_matrix(order, rank, dim_mode):
    adjm = np.diag(np.full((order-1,), rank), 1) if np.isscalar(rank) else np.diag(rank[:-1], 1)
    adjm[0, order-1] = rank if np.isscalar(rank) else rank[-1]
    adjm = adjm + adjm.transpose()
    np.fill_diagonal(adjm, dim_mode)

    return adjm

def index_to_onehot(indices, num_class):
    idx = np.asarray(indices) if isinstance(indices, list) else indices
    
    if idx.ndim != 2:
        raise ValueError("indices must be a 2D list or array")
    
    N, M = idx.shape

    one_hot_encoded = np.zeros((N, M, num_class), dtype=float)

    one_hot_encoded[np.arange(N)[:, None], np.arange(M), idx] = 1

    return one_hot_encoded


# Parameters
order_tensor = 4 # Order of the tensor
rank_tensor = 2 # Rank of the tensor
dim_tensor = 5 # Mode dimension
percentage_of_train = 1
percentage_of_test = 0
scale = 1

if percentage_of_train + percentage_of_test > 1:
    raise ValueError('The total percentage should be less than 1')

# Use NeuroTN to generate tensor
adjm = generate_TT_adj_matrix(order_tensor,rank_tensor,dim_tensor)
print('adjm_data:\n', adjm)


# Data generation 
def initialize_tensor(size, loc, scale):
    # Generate the tensor with exponential distribution
    tensor = np.random.normal(loc = loc, scale=scale, size=size)
    
    # Calculate the L1 norm of the tensor
    tensor = np.abs(tensor)
    l1_norm = np.sum(tensor)
    
    # Normalize the tensor to have an L1 norm of 1
    normalized_tensor = tensor / l1_norm
    
    return normalized_tensor

init_TN = functools.partial(initialize_tensor, loc=0.0, scale=scale)

output_dim = np.array([0] * (order_tensor-1)+[0])
# init_TN = functools.partial(np.random.exponential, scale=alpha)
DATA =  NeuroTN(adjm, output_dim, activation=lambda x:x, initializer = init_TN, core_mode=2)

# siz_data, siz_cores = compression_ratio(adjm, output_dim)

# print('siz_data:', siz_data)
# print('siz_cores:', siz_cores)

idx_data = [list(combo) for combo in itertools.product(range(dim_tensor), repeat=order_tensor)]
idx_onehot = index_to_onehot(idx_data, num_class=dim_tensor)
values = DATA.network_contraction(idx_onehot, return_contraction=True)

permuted_idx = np.random.permutation(idx_onehot.shape[0])
length_training = int(len(permuted_idx)*percentage_of_train)
length_test = int(len(permuted_idx)*percentage_of_test)

data_training = idx_onehot[permuted_idx[:length_training]]
values_training = values[permuted_idx[:length_training]]
data_test = idx_onehot[permuted_idx[length_training:(length_test+length_training)]]
values_test = values[permuted_idx[length_training:(length_test+length_training)]]

print(len(permuted_idx))
print(length_training)

print('data_test.shape', data_test.shape)
print('values_test.shape', values_test.shape)
print('Mean of data_train', jnp.mean(values_training))
print('Variance of data_train', jnp.var(values_training))
print(f"Min. value in training set: {np.min(values_training)}")
print(f"Max. value in training set: {np.max(values_training)}")
print(f"L1 norm of training data: {np.sum(values_training)}")


adjm_decomp = generate_TT_adj_matrix(order_tensor,rank_tensor,dim_tensor)
output_decomp = output_dim

print('adjm_decomp:\n', adjm_decomp)

print('========================')

TN = NeuroTN(adjm_decomp, output_decomp, activation=lambda x:x, initializer = init_TN, core_mode=2)
TN.target_shape = None



size_data = idx_onehot.shape[0]
batch_size = size_data
learning_rate = 0.3

loss_training = []
loss_test = []
eign_values = []
for epoch in range(10000):
    loss_training.append(TN.ABEG_iteration(learning_rate, data_training, values_training, verbose=True))
    if epoch % 10 == 0:
        print(type(data_test))
        if data_test.size == 0:
            print('Epoch: ', epoch, 'Training Loss: ', loss_training[-1], '; Testing Loss: N/A')
        else:
            predicted = TN.network_contraction(data_test, return_contraction=True)
            loss_test.append(np.mean(np.sum(np.square(predicted - values_test).reshape(predicted.shape[0],-1), axis=-1)))
            print('Epoch: ', epoch, 'Training Loss: ', loss_training[-1], '; Testing Loss: ', loss_test[-1])