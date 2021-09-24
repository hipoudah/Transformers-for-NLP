import numpy as np
from scipy.special import softmax

x = np.array([
    [1, 0, 1, 0],
    [0, 2, 0, 2],
    [1, 1, 1, 1]
])

w_query = np.array([
    [1., 0., 1.],
    [1., 0., 0.],
    [0., 0., 1.],
    [0., 1., 1.]
])

w_key = np.array([[0., 0., 1.],
                  [1., 1., 0.],
                  [0., 1., 0.],
                  [1., 1., 0.]])

w_value = np.array([[0, 2, 0],
                    [0, 3, 0],
                    [1, 0, 3],
                    [1, 1, 0]])

Q = np.matmul(x, w_query)
K = np.matmul(x, w_key)
V = np.matmul(x, w_value)
print("Q: ", Q)
print("V: ", V)
print("K: ", K)

d_k = 3
attention_scores = (Q @ K.transpose()) // int(np.sqrt(d_k))
attention_scores[0] = softmax(attention_scores[0])
attention_scores[1] = softmax(attention_scores[1])
attention_scores[2] = softmax(attention_scores[2])
print(attention_scores)

attention1 = attention_scores[0].reshape(-1, 1)
attention1 = attention_scores[0][0] * V[0]
attention2 = attention_scores[0][1] * V[1]
attention3 = attention_scores[0][2] * V[2]

attention_input1 = attention1 + attention2 + attention3

z0h1 = np.random.random((3, 64))
z1h2 = np.random.random((3, 64))
z2h3 = np.random.random((3, 64))
z3h4 = np.random.random((3, 64))
z4h5 = np.random.random((3, 64))
z5h6 = np.random.random((3, 64))
z6h7 = np.random.random((3, 64))
z7h8 = np.random.random((3, 64))
output_attention = np.hstack((z0h1, z1h2, z2h3, z3h4, z4h5, z5h6, z6h7, z7h8))
print(output_attention.shape)
