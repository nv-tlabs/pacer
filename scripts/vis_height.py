import joblib
import numpy as np
import matplotlib.pyplot as plt

 
heights = joblib.load("heights.pkl")
print(heights.min(), heights.max())
heights = heights.cpu().numpy()[0].reshape(-1, 3)
print(heights.shape)

plt.imshow(heights[:, 0].reshape(32, 32))
plt.show()
plt.imshow(heights[:, 1].reshape(32, 32))
plt.show()
plt.imshow(heights[:, 2].reshape(32, 32))
plt.show()