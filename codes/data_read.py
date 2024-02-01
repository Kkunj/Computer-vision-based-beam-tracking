import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import ast
from scipy.io import loadmat
import torch.nn.functional as F



def convert_mat(root):
    data = loadmat(root)['data'][:, 0] / 10
    return data

#print(x)
#print(type(x))

def create_samples(root_csv, portion=1.0, num_beam=64):
    f = pd.read_csv(root_csv)
    bbox_all = []
    beam_power_all = []
    lidar_all = []
    lidar_scr_all = []

    for idx, row in f.iterrows():
        # Process bbox data
        bbox_file_locations = row.loc["x_1":"x_8"]
        bboxes = []
        skip_row = False

        for file_location in bbox_file_locations:
            with open(str(file_location), 'r') as file:
                bbox_data = file.read()
                # Split the line into individual values
                bbox_values = [float(value) for value in bbox_data.split()[1:]]
                bbox_values = torch.tensor(bbox_values)
                # Check if the number of values is greater than 5
                if len(bbox_values) > 4:
                    skip_row = True
                    break
                bbox_resized = F.interpolate(bbox_values.view(1, 1, 1, 4), size=(64, 1), mode='nearest').squeeze()
                bboxes.append(np.asarray(bbox_resized))

        if skip_row:
            continue  # Skip processing this row

        bboxes = np.stack(bboxes, axis=0)
        bbox_all.append(bboxes)


        # # Lidar processing

        # lidar_file_location = row.loc["lidar_1":"lidar_8"]
        # lidar = []

        # for file_location in lidar_file_location:
        #     converted_file,size_of_file = convert_mat(file_location)
        #     converted_file = torch.Tensor(converted_file)
        #     lidar_resized = F.interpolate(converted_file.view(1, 1, 460, 2), size=(64, 1), mode='nearest').squeeze()
        #     lidar.append(lidar_resized)
        #     #print(size_of_file)

        # if skip_row:
        #     continue

        # lidar = np.stack(lidar,axis=0)
        # lidar_all.append(lidar)

        # Lidar SCR processing

        lidar_scr_file_location = row.loc["lidar_scr_1":"lidar_scr_8"]
        lidar_scr = []

        for file_location in lidar_scr_file_location:
            converted_scr_file = convert_mat(file_location)
            converted_scr_file = torch.tensor(converted_scr_file, requires_grad=False)
            # print(converted_scr_file.shape)

        if skip_row:
            continue

        lidar_scr = np.stack(lidar_scr,axis=0)
        lidar_scr_all.append(lidar_scr)


        



        # Process beam power data
        beam_power_file_locations = row.loc["y_1":"y_13"]
        beam_powers = []

        for file_location in beam_power_file_locations:
            with open(file_location, 'r') as file:
                beam_power_data = file.read()
                # Split the line into individual values
                
                beam_power_values = [float(value) for value in beam_power_data.split()[1:]]
                beam_powers.append(np.asarray([0.0] + beam_power_values))

        #bboxes = np.stack(bboxes, axis=0)
        #bbox_all.append(bboxes)


        if skip_row:
            continue  # Skip processing this row

        beam_powers = np.stack(beam_powers, axis=0)
        beam_power_all.append(beam_powers)

    bbox_all = np.stack(bbox_all, axis=0)
    bbox_all = torch.tensor(bbox_all)
    #print(bbox_all)
    print("bbox_all final shape:", bbox_all.shape)

    beam_power_all = np.stack(beam_power_all, axis=0)
    print("beam_power_all final shape:", beam_power_all.shape)

    # lidar_all = np.stack(lidar_all,axis=0)
    # print("lidar_all final shape: ",lidar_all.shape)
    # lidar_all = torch.tensor(lidar_all)

    lidar_scr_all = np.stack(lidar_scr_all,axis=0)
    print("lidar_scr_all final shape: ",lidar_scr_all.shape)
    lidar_scr_all = torch.tensor(lidar_scr_all)

    best_beam = np.argmax(beam_power_all, axis=-1)
    print("best_beam shape:", best_beam.shape)

    print("list is ready")
    num_data = len(beam_power_all)
    num_data = int(num_data * portion)





    stacked_data = torch.stack([bbox_all, lidar_scr_all], dim=1)
    # target_size = 59 * 8 * 3 * 64
    current_size = stacked_data.numel()
    print(current_size)

    # if current_size < target_size:
    # # If the size is not valid, repeat the input data to increase the size
    #     num_repeats = int(target_size / current_size) + 1
    #     stacked_data = torch.cat([stacked_data] * num_repeats, dim=0)

# Reshape the data to the appropriate shape
    # new_shape = (-1, 8, 3 * 64)
    # stacked_data = torch.reshape(stacked_data, new_shape)
    return stacked_data[:num_data], best_beam[:num_data, -5:]
    
    



    
# --------------------------here-----------------------------
class DataFeed(Dataset):
    def __init__(self, root_dir, portion=1.0, num_beam=64):

        self.root = root_dir
        self.samples, self.pred_val = create_samples(
            self.root, portion=portion, num_beam=num_beam
        )
        self.seq_len = 8
        self.num_beam = num_beam

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        samples = self.samples[idx]
        pred_val = self.pred_val[idx]

        samples = samples[-self.seq_len :]  # Read a sequence of tuples from a sample

        out_beam = torch.zeros((5,))
        bbox = torch.zeros((self.seq_len, 64))

        if not samples.size:
            samples = np.zeros(64)
        
        bbox = torch.tensor(samples, requires_grad=False)


        out_beam = torch.tensor(pred_val,requires_grad=False)
        return bbox.float(), out_beam.long()

if __name__ == "__main__":
   bbox,beam = create_samples("testing_data.csv")
   print(bbox.shape)
#    bbox = bbox.view(51,8,3,64)

   
#    reshaped_data = bbox.permute(0, 2, 1, 3)
#    reshaped_data = reshaped_data.contiguous().view(51, 8, 64)
#    print(reshaped_data.shape)
   print("-------------------")
   print(bbox.shape)
   print(beam.shape)
#    print(type(lidar))
#    print(type(lidar_scr))
