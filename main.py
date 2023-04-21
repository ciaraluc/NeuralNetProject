from typing import Tuple 
from neural import NeuralNet
from sklearn.model_selection import train_test_split

def parse_line(line:str)-> Tuple[List[float], List[float]]:
    tokens = line.split(",")
    out = int(tokens[36])

    inpt = [float(x) for x in tokens[0:35]]
    return (inpt, output)



with open("boneMarrowData.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

for line in training_data:
    print(line)