# The Data is Formatted Properly if this Program Succeeds

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

data, label = loadData('checkMe.txt')
print(label)
