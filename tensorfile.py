import numpy as np
from torch.utils.tensorboard import SummaryWriter

loss_list = []
with open("loss.txt", "r") as file1:
    all_lines = file1.readlines()
    loss_list = [l.strip() for l in all_lines]

crop_list = []
with open("crop.txt", "r") as file2:
    all_lines = file2.readlines()
    crop_list = [l.strip() for l in all_lines]
#accuracy_list = []
#with open("accuracy.txt", "r") as file2:
#    all_lines = file2.readlines()
#    accuracy_list = [l.strip() for l in all_lines]

# define writer for tensorboard implementation
writer = SummaryWriter()

# tensor board information
for i in range(0, len(loss_list)):
    writer.add_scalar("Loss", float(loss_list[i]), i)

for i in range(0, len(crop_list)):
    writer.add_scalar("Cropped", float(crop_list[i]), i)
writer.close()
