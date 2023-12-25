import jax
import torch
import torch.nn as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt


n = 200
layers = [1,10,10,10,1]
lr = 0.001
epochs = 300
stddev = 0.3

key = jax.random.PRNGKey(42)
key, xkey, ynoisekey = jax.random.split(key,3)
x_data = jax.random.uniform(xkey,(n,1), minval=0.0,maxval=2*jnp.pi)
y_data = jnp.cos(x_data) + jax.random.normal(ynoisekey,(n,1)) * stddev



weights,biases,actvns = [],[],[]

for (i,j) in zip(layers[:-1],layers[1:]):
    kernel_uniform = jnp.sqrt(6/(i+j))
    key,wkey = jax.random.split(key)
    w = jax.random.uniform(wkey,(i,j),
                           minval=-kernel_uniform,
                           maxval=+kernel_uniform)
    b = jnp.zeros(j)
    
    weights.append(w)
    biases.append(b)
    actvns.append(jax.nn.sigmoid)
actvns[-1] = lambda x:x

def forward_pass(x,weights,biases,activations):
    a = x
    
    for w,b,f in zip(weights,biases,activations):
        a = f(a @ w + b)
        
    return a

preds = forward_pass(x_data,weights,biases,actvns)
plt.scatter(x_data,y_data,color='green')
plt.scatter(x_data,preds,color='red')
plt.show()

def find_loss(y_pred,y_gt):
    loss = 0.5 * (jnp.mean((y_pred-y_gt)**2))
    return loss
 
out = find_loss(preds,y_data)
print(out)
