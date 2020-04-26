# Split Training Data Script
# This script randomly splits the master training set into training and testing data
# Output --> TrainingData.txt    &    TestDataRaw.txt    &    TestDataLbl.txt

from sklearn.model_selection import train_test_split

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

def exportTrData(data, label, fname):
    file = open(fname, "w")
    for i in range(len(data)):
        file.write(str(data[i]) + '\t' + str(label[i]) + '\n')
    file.close()

def exportTeData(data, fname):
    file = open(fname, "w")
    for line in data:
        file.write(str(line) + '\n')
    file.close()

# Load in Matser Training Dataset
data, label = loadData('MasterTrainData.txt')

# Split Data into 80% Training & 20% Testing Data
trData, teData, trLabl, teLabl = train_test_split(data, label, test_size = 0.20, random_state = 42)

# Create 3 New Files
exportTrData(trData, trLabl, 'TrainingData.txt')
exportTeData(teData, 'TestDataRaw.txt')
exportTeData(teLabl, 'TestDataLbl.txt')