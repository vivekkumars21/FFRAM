import numpy as np
import seaborn as sns

arr = np.array([-233,1,23,5,6,7,8,5,3,2,44,11])


q1 = np.percentile(arr,25)
q3 = np.percentile(arr,75)
iqr = q3 -q1

uf = q3 + (1.5*iqr)
lf = q1 - (1.5*iqr)

l = []

for i in arr:
    if i >= lf and i <= uf:
        l.append(i)
arr2 = np.array(l)

sns.boxplot(x = arr2)
