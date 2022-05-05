import random

with open("all.csv", "r") as f:
    lines = f.readlines()

header = lines[0]
lines = lines[1:]
total_lines = len(lines)

indices = range(total_lines)
test_indices = random.sample(indices, 2000)
test_set = set(test_indices)
train_indices = [line for line in indices if line not in test_set]

with open("train.csv", "w") as train:
    train.write(header)
    for line in train_indices:
        train.write(lines[line])

with open("val.csv", "w") as val:
    val.write(header)
    for line in test_indices:
        val.write(lines[line])

