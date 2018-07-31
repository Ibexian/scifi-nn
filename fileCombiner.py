import glob
import re


path='/Users/wkamovitch/Sites/scifinn/input_data/Sci-Fi Texts/*.txt'

files= glob.glob(path)

combinedTrain = open('combined.train.txt', 'w')
combinedVerify = open('combined.valid.txt', 'w')
combinedTest = open('combined.test.txt', 'w')

for index, file in enumerate(files):
    f=open(file, 'r')
    print(file)
    content = f.read()
    if index % 3 == 0:
        combinedTrain.write(content)
    if index % 7 == 0:
        combinedTest.write(content)
    if index % 5 == 0:
        combinedVerify.write(content)

    f.close()


combinedTrain.close()
combinedVerify.close()
combinedTest.close()

