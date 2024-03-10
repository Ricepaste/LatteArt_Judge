import numpy as np

num = np.array([3, 2, 5])
num = num.reshape(1, 3)

# trans = np.array([[0.0000, 0.4286, 0.5714],
#                   [1.0000, 0.0000, 0.0000],
#                   [0.2500, 0.7500, 0.0000]])

trans = np.array([[0.0000, 0.6666, 0.3334],
                  [0.5000, 0.0000, 0.5000],
                  [0.2000, 0.8000, 0.0000]])
while True:
    num = num @ trans
    print(num)
