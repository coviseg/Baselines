import matplotlib.pyplot as plt
import numpy as np

array = np.loadtxt("/scratch/coviseg/teste.txt")
plt.plot(range(len(array)), array)
plt.savefig("/scratch/coviseg/teste/outtest.png")


