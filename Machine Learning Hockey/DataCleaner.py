import csv

cleanedrows = []
with open("ML_10y_playerdata.csv", r, newline='') as file:
    reader = csv.reader(file)
    for line in reader:
        if line[8] != '-':
            try:
                pos_int = int(line[2])
                cleanedrows.append(line)
            except ValueError:
                None
