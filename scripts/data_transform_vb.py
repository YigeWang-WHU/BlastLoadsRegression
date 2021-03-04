import csv
import glob

# Read all csv data files
data = []
for f in glob.glob("raw_data/values.csv"):
    data.append(f)


# Processing
for f in data:
    name = f.split('/')[-1].split('.')[-2]
    with open(f, 'r', encoding='utf-8', errors='ignore') as d:
        reader = list(csv.reader(d))
        reader = reader[1:]  # Remove titles
        reader = [r[0:2] + r[9:16] for r in reader]  # Remove redundant columns
        with open("raw_data/" + name + "_SB.txt",
                  'w') as new_txt:  # Write to new CSV
            for row in reader:
                new_txt.write(str(row) + '\n')

