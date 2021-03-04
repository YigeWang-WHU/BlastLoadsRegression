import csv
import glob

# Read all csv data files
data = []
for f in glob.glob("raw_data/data_FF.csv"):
    data.append(f)


# Processing
for f in data:
    name = f.split('/')[-1].split('.')[-2]
    with open(f, 'r') as d:
        reader = list(csv.reader(d))
        reader = reader[2:]  # Remove titles
        reader = [r[0:5] for r in reader]  # Remove redundant columns
        with open("raw_data/" + name + "_transformed.txt",
                  'w') as new_txt:  # Write to new CSV
            for row in reader:
                new_txt.write(str(row) + '\n')

