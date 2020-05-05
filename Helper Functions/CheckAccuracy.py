def loadData(fname):
    labels = []
    file = open(fname)
    for line in file:
        rating = line.strip() 
        labels.append(int(rating))
    file.close()
    return labels

def checkScore(data1, data2):
    i = 0
    correct = 0
    for x in range(len(data1)):
        i += 1
        if data1[x] == data2[x]:
            correct += 1
    return correct/i

# Load in Matser Training Dataset
trueLabel = loadData('TestDataLbl.txt')
predLabel = loadData("OUTPUT-LabeledTestData.txt")
print(checkScore(trueLabel, predLabel))