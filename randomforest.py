from sklearn.ensemble import RandomForestClassifier

import csv as csv, numpy as np

csv_file_object = csv.reader(open('./train_sk.csv', 'rb'))

train_header = csv_file_object.next() # skip the header

train_data = []

for row in csv_file_object:
    train_data.append(row)
    
train_data = np.array(train_data)

Forest = RandomForestClassifier(n_estimators = 100)

Forest = Forest.fit(train_data[0::, 1::], train_data[0::, 0])

test_file_object = csv.reader(open('./test_sk.csv', 'rb'))

test_header = test_file_object.next() # skip header row

test_data = []

for row in test_file_object:
    test_data.append(row)
    

test_data = np.array(test_data)

output = Forest.predict(test_data)

output = output.astype(int)

np.savetxt('./output.csv', output, delimiter=',') 










