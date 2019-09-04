import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
c = -np.log(1/(1+np.exp(-x)))



plt.plot(x, c)
plt.show()

