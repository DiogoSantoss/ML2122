import numpy as np

w_1 = np.array([[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1]])
w_2 = np.array([[1,1,1],[1,1,1]])
w_3 = np.array([[0,0],[0,0]])

b_1 = np.array([[1],[1],[1]])
b_2 = np.array([[1],[1]])
b_3 = np.array([[0],[0]])

x_0 = np.array([[1],[1],[1],[1],[1]])
z_0 = np.array([[1],[0]])

def forward_propagation(w,x,b):
    z = np.dot(w,x) + b
    x_new = np.array([np.tanh(zi) for zi in z])
    return z,x_new

z_1,x_1 = forward_propagation(w_1,x_0,b_1)
z_2,x_2 = forward_propagation(w_2,x_1,b_2)
z_3,x_3 = forward_propagation(w_3,x_2,b_3)

print(z_1)
print(x_1)
print(z_2)
z_new = np.array([1-np.tanh(zi)**2 for zi in z_2])
print(z_new)
print(x_2)
print(z_3)
print(x_3)


def back_propagation(x,z):

    dXdZ = np.array([(1-np.tanh(zi))**2 for zi in z])
    dEdX = x-z
    dEdZ = dEdX * dXdZ

    return dEdZ

#dEdZ_3 = back_propagation(x_3,z_3)
#print(dEdZ_3)

