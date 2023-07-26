import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import linear_model

# y = mx + c
# F = 1.8*C + 32

x = list(range(0, 10))
y = [1.8 * F + 32 for F in x]

print(f'X: {x}')
print(f'Y: {y}')

plt.plot(x, y, '-*b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Points')
plt.show()

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x, y, test_size=0.25, random_state=42)

model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

accuracy = model.score(xTest, yTest)
print("Accuracy Score:", accuracy*100)
