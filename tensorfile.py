import numpy as np
from torch.utils.tensorboard import SummaryWriter

loss_list = []
file_name1 = "loss.txt"
file1 = open(file_name1, "r")
for line in file1:
    loss_list.append(line.strip())
file1.close()

accuracy_list = []
file_name2 = "accuracy.txt"
file2 = open(file_name2, "r")
for line in file2:
    accuracy_list.append(line.strip())
file2.close()

# define writer for tensorboard implementation
writer = SummaryWriter()

# tensor board information
for i in range(0, len(loss_list)):
    writer.add_scalar("Loss", float(loss_list[i]), i)
writer.close()

for i in range(0, len(accuracy_list)):
    writer.add_scalar("Accuracy", float(accuracy_list[i]), i)
