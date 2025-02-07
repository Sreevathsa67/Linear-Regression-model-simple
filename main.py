import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([70, 40, 80, 50, 70, 60, 20, 10, 80, 60, 40, 90, 50, 30, 30]).reshape(-1, 1)
scores = np.array([75, 49, 91, 51, 74, 68, 20, 8, 87, 74, 50, 96, 58, 32, 33]).reshape(-1, 1)

time_train, time_test,score_train,score_test = train_test_split(time_studied,scores,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(time_train,score_train)
print(model.score(time_test,score_test))
print(model.predict(np.array([40]).reshape(-1,1)))
plt.scatter(time_studied,scores)

plt.plot(np.linspace(0, 70, 100).reshape(-1, 1),
         model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r')

plt.ylim(0, 100)
plt.show()