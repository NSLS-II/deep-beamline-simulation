import numpy as np
from torch.utils.tensorboard import SummaryWriter

loss_list = []
with open("loss.txt", 'r') as file1:
    line = file1.readline()
    loss_list.append(line.strip())

accuracy_list = []

with open("accuracy.txt", "r") as file2:
    line = file2.readline()
    accuracy_list.append(line.strip())

# define writer for tensorboard implementation
writer = SummaryWriter()

# tensor board information
for i in range(0, len(loss_list)):
    writer.add_scalar("Loss", float(loss_list[i]), i)
writer.close()

for i in range(0, len(accuracy_list)):
    writer.add_scalar("Accuracy", float(accuracy_list[i]), i)
