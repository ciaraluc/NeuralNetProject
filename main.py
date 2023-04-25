from typing import Tuple, List
from neural import NeuralNet
from sklearn.model_selection import train_test_split

def parse_line(line:str)-> Tuple[List[float], List[float]]:
    tokens = line.split(",")
    out = [int(tokens[36])]
    print(out)
    inpt = [float(x) for x in tokens[0:35]]
    return (inpt, out)

def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data

with open("boneMarrowData.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

for line in training_data:
    print(line)

td = normalize(training_data)

print()
print("normalize")
print(td)

train_data, test_data=train_test_split(td)
# print(train_data)
nn = NeuralNet(35, 50, 1)
nn.train(train_data, iters=1000, print_interval=1000, learning_rate=0.1)

print(nn.test_with_expected(test_data))

