# Seperate Test Data Script
# This script removes labels from testing data and places them into a seperate file
# Output --> TestDataRaw.txt    &    TestDataLbl.txt

def loadData(fname):
    reviews = []
    labels = []
    file = open(fname)
    for line in file:
        review, rating = line.strip().split('\t')  
        reviews.append(review.lower())    
        labels.append(int(rating))
    file.close()
    return reviews,labels

def exportData(data, fname):
    file = open(fname, "w")
    for line in data:
        file.write(str(line) + '\n')
    file.close()

teData, teLabel = loadData('TestData.txt')
exportData(teData , 'TestDataRaw.txt')
exportData(teLabel, 'TestDataLbl.txt')