import csv
import glob
import scipy.io
# Read all csv data files
data = []
for f in glob.glob("raw_data/data_tot.mat"):
    data.append(f)


# Processing
for f in data:
    name = f.split('/')[-1].split('.')[-2]
    mat = scipy.io.loadmat(f)
    mat = mat['data_tot']
    mat = mat.tolist()
    with open("raw_data/" + name + "_mat.txt", 'w') as r:
        for p in mat:
            r.write(str(p) + '\n')
        		


