__author__ = "Akshay Raman"

import sys
import pandas as pd
import numpy as np
import scipy.stats as stats

if len(sys.argv) < 2:
    print("Missing file name")

fileName = sys.argv[1]
dataset = pd.read_csv(fileName)

#Contains all sensor readings
data = dataset.iloc[:, 1:].values
labels = ["walking", "standing", "sitting", "lying down"]
axes = ["x", "y", "z"]
fn_list = [np.min, np.max, np.mean, stats.mode, np.median, np.var, np.std, stats.iqr]
fn_desc = ["min", "max", "mean", "mode", "median", "var", "sd", "iqr"]

headings = []
result_list = [[] for i in range(len(labels))]
set_header=True

for i, label in enumerate(labels):
    if i!=0:
        set_header=False
    #print(label)
    label_data = data[:, 1:][data[:,0]==i]
    for j, axis in enumerate(axes):
        for k,f in enumerate(fn_list):
            val = f(label_data[:,j])
            if fn_desc[k] == "mode":
                val = val[0][0]
            desc = "%s(%s)" %(fn_desc[k], axis)

            result_list[i].append(val)
            if set_header:
                headings.append(desc)
            #print("%s %s" %(desc, val))

    for j, axis in enumerate(axes):
        x = label_data[:,j]
        rms_val = np.sqrt(x.dot(x)/x.size)

        result_list[i].append(rms_val)
        desc = "%s(%s)" %("rms", axis)
        if set_header:
            headings.append(desc)
        #print("%s %s" %(desc, rms_val))

    for j, axis in enumerate(axes):
        x = label_data[:,j]
        arr = np.ma.array(x).compressed()
        mad = np.mean(arr)
        mad_val = np.mean(np.abs(arr - mad))
        result_list[i].append(mad_val)
        desc = "%s(%s)" %("mad", axis)
        if set_header:
            headings.append(desc)
        #print("%s %s" %(desc, mad_val))


    axes_sum = label_data.sum(axis=1).mean()
    #print("mean of sum across axes", axes_sum)
    for _ in range(3):
        a,b = _, (_+1)%3
        coef = stats.pearsonr(label_data[:,a], label_data[:,b])[0]
        result_list[i].append(coef)
        desc = "corr(%s, %s)" %(axes[a], axes[b])
        if set_header:
            headings.append(desc)
        #print("%s %s" %(desc, coef))


_display = [""]+labels
final_data = [headings] + result_list
for i,j in enumerate(final_data):
    if i>0:
        r = ['{0:.2f}'.format(float(t)) for t in final_data[i]]
        print("%s\t%s" %(_display[i], '\t'.join(r)))
    else:
        print("%s\t%s" %(_display[i], '\t'.join(final_data[i])))


