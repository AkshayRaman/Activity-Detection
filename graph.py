__author__ = "Akshay Raman"

import pandas as pd
import matplotlib.pyplot as plt
import sys

activity = sys.argv[1]

dataset = pd.read_csv(activity+'.txt', header=None)
data = dataset.iloc[:, 2:].values

legend = ["linear_acceleration", "gravity", "accelerometer",
        "gyroscope", "rotation_vector", "orientation"]
axis = ["x","y","z"]*4 + ["x*sin(theta/2)", "y*sin(theta/2)", "z*sin(theta/2)", "cos(theta/2)", "heading"] + ["azimuth", "pitch", "roll"]
count = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14,15,16],[17,18,19]]


x=1000

fig, axes = plt.subplots(2, 3)
#plt.suptitle(activity)
for i,label in enumerate(legend):
    axes[i//3, i%3].set_title(label)
    axes[i//3, i%3].set_ylabel("Sensor reading")

    indices = count[i]
    for j in indices:
        axes[i//3,i%3].plot(range(x), data[:,j][:x], label = axis[j])
        axes[i//3,i%3].legend(loc='best')

plt.show()
#plt.savefig("%s.png" %activity, dpi=600)
