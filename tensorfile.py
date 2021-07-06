import numpy as np
from torch.utils.tensorboard import SummaryWriter

loss_list = []
file_name = "tensorboard.txt"
file = open(file_name, "r")
for line in file:
    loss_list.append(line.strip())
file.close()

# define writer for tensorboard implementation
writer = SummaryWriter()

# tensor board information
for i in range(0, len(loss_list)):
    writer.add_scalar("Loss", float(loss_list[i]), i)
writer.close()
