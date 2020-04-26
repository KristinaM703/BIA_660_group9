from csv import reader

with open('inputData.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    newFile = open('newFile.txt', "w")
    for row in csv_reader:
        newFile.write(str(row[0]) + '\n')
