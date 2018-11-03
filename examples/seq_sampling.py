import numpy as np
import matplotlib.pyplot as plt

n = 10000
k = 10
batch_size = 1000

r = np.random.geometric(p=(k/np.exp(k)) / n, size=[batch_size, k])
#r = np.random.negative_binomial(n, (k/np.exp(k)) / n, size=[batch_size, k + 10])

cs = np.cumsum(r, axis=-1) - 1

max = np.argmax(cs, axis=-1)

rotation = np.random.uniform(size=[batch_size])
rotation = np.multiply(rotation, max)
rotation = np.expand_dims(rotation, axis=-1)
max = np.expand_dims(max, axis=-1)

print(np.shape(max))
print(np.shape(rotation))

s = (cs + rotation) % max

s = s[..., :k]

plt.hist(s)
plt.show()

unique = np.array(list(map(lambda x: len(np.unique(x)), s)))

plt.hist(unique)
plt.show()
