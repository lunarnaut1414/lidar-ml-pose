## Import Essential Libraries
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
import matplotlib.pyplot as plt
import string
import scipy.io as sio
import scipy.linalg as slinalg
import torchvision
from torchvision import datasets, transforms
import glob
from scipy.io import savemat
from torch.autograd import Variable

# Set random seed
np.random.seed(0)

# Constants
RAD_TO_DEG = 180 / np.pi

# Enable gpu processing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Loading Data
# Load the Lidar Data Log
# lidarDataLog = sio.loadmat('./lidar_images_cart/Lidar_Image_Double_Loop_Cart_1.mat')
lidarDataLog = sio.loadmat('./lidar_images_cart_cov/Lidar_Image_Double_Loop_Cart_1.mat')

## Fetch the number of double loop files
# trajectory_files = glob.glob("./lidar_images_cart/*.mat")
trajectory_files = glob.glob("./lidar_images_cart_cov/*.mat")
num_files = trajectory_files.__len__()

# Lidar Point Cloud Regularization Factor
lidarRegFactor = 1000

# Azimuth Angle Regularization Factor
azRegFactor = 1

lidarDataLog = sio.loadmat(trajectory_files[0])

# Pull the 2D lidar image [input]
lidarCart = lidarDataLog['log']['lidarCart'][0][0]
lidarCart = torch.tensor(np.float32(lidarCart))

# Pull the position of the vehicle in the nav frame [target]
posVehInNav = lidarDataLog['log']['posVehInNav'][0][0]
posVehInNav = torch.tensor(np.float32(posVehInNav))

# Pull the attitude of the vehicle in the nav frame [target]
azVehInNav = lidarDataLog['log']['azVehInNav'][0][0]
azVehInNav = torch.tensor(np.float32(azVehInNav)) / azRegFactor

# Pull the covariance of the estimation [target]
covPoseInNav = lidarDataLog['log']['covariance'][0][0]
covPoseInNav = torch.tensor(np.float32(covPoseInNav))

num_train = 30
posVehInNavLog = torch.zeros(tuple(torch.cat((torch.tensor([num_train]), torch.tensor(posVehInNav.shape)), axis=0)))
azVehInNavLog = torch.zeros(tuple(torch.cat((torch.tensor([num_train]), torch.tensor(azVehInNav.shape)), axis=0)))
covPoseInNavLog = torch.zeros(tuple(torch.cat((torch.tensor([num_train]), torch.tensor(covPoseInNav.shape)), axis=0)))
lidarCartLog = torch.zeros(tuple(torch.cat((torch.tensor([num_train]), torch.tensor(lidarCart.shape)), axis=0)))

for file_idx in range(num_train):
    # Loading Data
    # Load the Lidar Data Log
    lidarDataLog = sio.loadmat(trajectory_files[file_idx])

    # Pull the 2D lidar image [input]
    lidarCart = lidarDataLog['log']['lidarCart'][0][0]
    lidarCart = torch.tensor(np.float32(lidarCart))

    # Pull the position of the vehicle in the nav frame [target]
    posVehInNav = lidarDataLog['log']['posVehInNav'][0][0]
    posVehInNav = torch.tensor(np.float32(posVehInNav)) / lidarRegFactor

    # Pull the attitude of the vehicle in the nav frame [target]
    azVehInNav = lidarDataLog['log']['azVehInNav'][0][0]
    azVehInNav = torch.tensor(np.float32(azVehInNav)) / azRegFactor

    # Pull the covariance of the estimation [target]
    covPoseInNav = lidarDataLog['log']['covariance'][0][0]
    covPoseInNav = torch.tensor(np.float32(covPoseInNav))

    # Save the data into a log tensor
    lidarCartLog[file_idx,:,:,:] = lidarCart
    posVehInNavLog[file_idx,:,:, ] = posVehInNav
    azVehInNavLog[file_idx,:,:] = azVehInNav
    covPoseInNavLog[file_idx,:,:, :] = covPoseInNav

## Define the training and validation index
# Index all of the images
numLidarImg = lidarCart.shape[0]
lidarImgIdx = np.arange(numLidarImg)

# Shuffle the index
# np.random.shuffle(lidarImgIdx)

# Define the training and valid set size
numTrainImg = int( np.ceil( 0.8 * numLidarImg ) )
numValidImg = int( np.ceil( 0.2 * numLidarImg ) )

# Define the set of training and valid images
train_idx = lidarImgIdx[0:int( np.ceil( numLidarImg ) )]

# Validate the training and validation set size
print(numTrainImg)
print(numValidImg)

# Define the Lidar CNN
class Lidar_CNN_Pose_Cov(nn.Module):
    def __init__(self):
        super(Lidar_CNN_Pose_Cov, self).__init__()

        # Define constants
        self.image_size = np.uint16([60, 200])
        self.num_fc_input = int(2800)
        self.out_pose_size = 3
        self.out_cov_size = 6
        self.out_diag_size = 3
        self.kernel_size = np.uint16([2,2])
        self.num_conv_layer = 3
        self.max_pool_stride = np.uint16([2, 2])

        # Define the convolutional layers
        self.en_conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=[2, 2], stride=[1, 1])
        self.en_conv2 = nn.Conv2d(in_channels=self.en_conv1.out_channels, out_channels=self.en_conv1.out_channels * 2, kernel_size=[2, 2], stride=[1, 1])
        self.en_conv3 = nn.Conv2d(in_channels=self.en_conv2.out_channels, out_channels=self.en_conv2.out_channels * 2, kernel_size=[2, 2], stride=[1, 1])

        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size = [1,1], stride=[2,2])

        # Define the fully connected layers
        self.fc1_pose = nn.Linear(self.num_fc_input, int(self.num_fc_input/2))
        self.fc2_pose = nn.Linear(self.fc1_pose.out_features, int(self.fc1_pose.out_features/2))
        self.fc3_pose = nn.Linear(self.fc2_pose.out_features, int(self.fc2_pose.out_features/2))
        self.fc4_pose = nn.Linear(self.fc3_pose.out_features, int(self.fc3_pose.out_features/2))

        self.fc1_cov = nn.Linear(self.num_fc_input, int(self.num_fc_input / 2))
        self.fc2_cov = nn.Linear(self.fc1_cov.out_features, int(self.fc1_cov.out_features / 2))
        self.fc3_cov = nn.Linear(self.fc2_cov.out_features, int(self.fc2_cov.out_features / 2))

        # Define the dropoout layer
        self.dropout = nn.Dropout(p = 0.05)

        # Define the position and attitude output layer
        # self.out_pose = nn.Linear(self.fc3_pose.out_features, self.out_pose_size)
        self.out_pose = nn.Linear(self.fc4_pose.out_features, self.out_pose_size)
        self.out_cov_l = nn.Linear(self.fc3_cov.out_features, self.out_cov_size)

        # Define the neural network structure
    def forward(self, x):
        # 1st convolution Layer and max pooling
        x = self.pool(self.en_conv1(x))
        # Save the first convolution image
        conv1_img = x

        # 2nd convolution Layer and max pooling
        x = self.pool(self.en_conv2(x))
        # Save the first convolution image
        conv2_img = x

        # 3rd convolution Layer and max pooling
        x = self.pool(self.en_conv3(x))
        # Save the first convolution image
        conv3_img = x

        # Flatten the convolution
        x = x.view(-1, self.num_fc_input)

        # 1st Dropout Layer
        x_pose = self.dropout(x)
        x_cov  = self.dropout(x)
        # 1st Fully Connected Layer
        x_pose = F.relu(self.fc1_pose(x_pose))
        x_cov  = F.relu(self.fc1_cov(x_cov))

        # 2nd Dropout Layer
        x_pose = self.dropout(x_pose)
        x_cov  = self.dropout(x_cov)
        # 2nd Fully Connected Layer
        x_pose = F.relu(self.fc2_pose(x_pose))
        x_cov  = F.relu(self.fc2_cov(x_cov))

        # 3rd Dropout Layer
        x_pose = self.dropout(x_pose)
        x_cov  = self.dropout(x_cov)
        # 3rd Fully Connected Layer
        x_pose = F.relu(self.fc3_pose(x_pose))
        x_cov  = F.relu(self.fc3_cov(x_cov))

        # 4th Dropout Layer
        x_pose = self.dropout(x_pose)
        x_pose = F.relu(self.fc4_pose(x_pose))

        # Output Layer
        pose_pred = self.out_pose(x_pose)
        cov_pred  = self.out_cov_l(x_cov)
        return pose_pred, cov_pred

def pose_norm_loss(pose_pred, pose_target, cov_est, num_train ):
    I = torch.eye(3)
    loss = 0

    for i in range(num_train):
        pose_error = pose_target[:, i] - pose_pred[:, i]
        loss_j = torch.matmul(torch.matmul(pose_error, torch.linalg.inv(cov_est)), pose_error) / 1000 / 2

        # Add loss for each covariance estimation
        loss += loss_j

    # Return the loss
    return loss

def cov_cholesky_norm_loss(cov_mat_target, cov_mat_pred_L, num_train ):
    I = torch.eye(3)
    loss = 0

    for i in range(num_train):
        # Calculate the loss
        cov_target = cov_mat_target[i, :, :]
        L_target = torch.linalg.cholesky(cov_target)
        L_pred = cov_mat_pred_L[i, :, :]

        cov_pred = torch.matmul(L_pred, L_pred.T)
        cov_ratio = torch.matmul(torch.matmul(torch.linalg.inv(L_target.T), cov_pred), torch.linalg.inv(L_target))

        cov_pred_error = torch.matmul((I - cov_ratio), (I - cov_ratio).T)
        loss_j = 1/2 * torch.sqrt(torch.trace(cov_pred_error)) / 10000

        # Add loss for each covariance estimation
        loss += loss_j

    # Return the loss
    return loss

# Initialize the LIDAR CNN
lidarCNNPoseCov = Lidar_CNN_Pose_Cov().cuda()
# lidarCNNPose    = Lidar_CNN_Pose().cuda()

# Define the loss function and optimizer
criterion       = nn.MSELoss()
criterion_pose  = nn.L1Loss()
criterion_cov   = nn.L1Loss()

train_cov = True
if (train_cov == True):
    lidarCNNPoseCov.load_state_dict(torch.load('./model/cartesian/lidarCNN_Cart_Pose10000.pth'))

    # Freeze the convolutional layers
    lidarCNNPoseCov.en_conv1.weight.requires_grad = False
    lidarCNNPoseCov.en_conv1.bias.requires_grad = False
    lidarCNNPoseCov.en_conv2.weight.requires_grad = False
    lidarCNNPoseCov.en_conv2.bias.requires_grad = False
    lidarCNNPoseCov.en_conv3.weight.requires_grad = False
    lidarCNNPoseCov.en_conv3.bias.requires_grad = False

    # Freeze the layers used to calculate the pose
    lidarCNNPoseCov.fc1_pose.weight.requires_grad = False
    lidarCNNPoseCov.fc1_pose.bias.requires_grad = False
    lidarCNNPoseCov.fc2_pose.weight.requires_grad = False
    lidarCNNPoseCov.fc2_pose.bias.requires_grad = False
    lidarCNNPoseCov.fc3_pose.weight.requires_grad = False
    lidarCNNPoseCov.fc3_pose.bias.requires_grad = False
    lidarCNNPoseCov.out_pose.weight.requires_grad = False
    lidarCNNPoseCov.out_pose.bias.requires_grad   = False

    # Unfreeze the layers used to calculate the covariance
    lidarCNNPoseCov.fc1_cov.weight.requires_grad = True
    lidarCNNPoseCov.fc1_cov.bias.requires_grad   = True
    lidarCNNPoseCov.fc2_cov.weight.requires_grad = True
    lidarCNNPoseCov.fc2_cov.bias.requires_grad   = True
    lidarCNNPoseCov.fc3_cov.weight.requires_grad = True
    lidarCNNPoseCov.fc3_cov.bias.requires_grad   = True
    lidarCNNPoseCov.out_cov_l.weight.requires_grad = True
    lidarCNNPoseCov.out_cov_l.bias.requires_grad   = True

else:
    # Freeze the layers used to calculate the covariance
    lidarCNNPoseCov.fc1_cov.weight.requires_grad = False
    lidarCNNPoseCov.fc1_cov.bias.requires_grad   = False
    lidarCNNPoseCov.fc2_cov.weight.requires_grad = False
    lidarCNNPoseCov.fc2_cov.bias.requires_grad   = False
    lidarCNNPoseCov.fc3_cov.weight.requires_grad = False
    lidarCNNPoseCov.fc3_cov.bias.requires_grad   = False
    lidarCNNPoseCov.out_cov_l.weight.requires_grad = False
    lidarCNNPoseCov.out_cov_l.bias.requires_grad   = False

# Setup the optimizer and scheduler
optimizer = optim.SGD(lidarCNNPoseCov.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

alpha = 0.01
beta = 10
gamma = 1

cov_est = torch.tensor([(10 / lidarRegFactor) ** 2, (10 / lidarRegFactor) ** 2, (1 / RAD_TO_DEG / azRegFactor) ** 2])
cov_est = torch.diag(cov_est)

training_enable = 1
save_model = 1

## Training Section
if (training_enable == True):
    num_epoch = 10000

    for epoch in range(0,num_epoch+1):
        running_loss = 0

        for img_idx in train_idx:
            # Pull the input tensor
            input_tensor = lidarCartLog[:, img_idx, :, :, :]
            # input_tensor = input_tensor.unsqueeze(1)

            # Pull the target tensors
            pos_target = posVehInNavLog[:, :, img_idx]
            att_target = azVehInNavLog[:, :, img_idx]
            pose_target = torch.concat([pos_target,att_target], axis=1)

            cov_mat_target = covPoseInNavLog[:,img_idx,:,:]

            # Run the lidar image through the lidar CNN
            pose_pred, cov_pred = lidarCNNPoseCov(input_tensor.to(device))
            pose_pred = Tensor.cpu(pose_pred)
            cov_pred  = Tensor.cpu(cov_pred)

            # Convert the covariance prediction into a matrix
            cov_mat_pred_L = torch.zeros(num_train, 3, 3)
            cov_mat_pred_D = torch.zeros(num_train, 3, 3)
            cov_mat_pred_L[:, 0, 0] = cov_pred[:, 0]
            cov_mat_pred_L[:, 1, 0] = cov_pred[:, 1]
            cov_mat_pred_L[:, 1, 1] = cov_pred[:, 2]
            cov_mat_pred_L[:, 2, 0] = cov_pred[:, 3]
            cov_mat_pred_L[:, 2, 1] = cov_pred[:, 4]
            cov_mat_pred_L[:, 2, 2] = cov_pred[:, 5]

            cov_mat_pred_D[:, 0, 0] = 1
            cov_mat_pred_D[:, 1, 1] = 1
            cov_mat_pred_D[:, 2, 2] = 1

            cov_mat_pred = torch.zeros(num_train, 3, 3)
            for i in range(num_train):
                cov_mat_pred[i,:,:] = torch.matmul( torch.matmul(cov_mat_pred_L[i,:,:], cov_mat_pred_D[i,:,:]), cov_mat_pred_L[i,:,:].T.squeeze())

            pos_pred = pose_pred[:,0:2]
            att_pred = pose_pred[:, 2]

            # Concatenate the target
            # target = torch.cat((pos_target, att_target, cov_target), axis=1)

            # Compute the estimation loss
            # loss_pose = criterion_pose(pose_pred,pose_target)

            if (train_cov == True):
                loss = cov_cholesky_norm_loss(cov_mat_target, cov_mat_pred_L, num_train)
            else:
                loss = pose_norm_loss(pose_pred.T, pose_target.T, cov_est, num_train)

            loss.backward()

            # Optimize
            optimizer.step()
            optimizer.zero_grad()

            # Debugging
            running_loss += loss.item()
        # End of Iteration Loop

        # Break training if loss fall belows some threshold
        # if (epoch % 1000 == 0):
        #     scheduler.step()

        # Print Results
        if epoch % 1 == 0:
            print("Epoch: %d, loss: %1.10f" % (epoch, running_loss))
        if epoch % 250 == 0:
            print('Saving model!')
            if (train_cov == False):
                file_name = 'lidarCNN_Cart_Pose' + str(epoch)
            else:
                file_name = 'lidarCNN_Cart_PoseCov' + str(epoch)
            torch.save(lidarCNNPoseCov.state_dict(), './model/cartesian/'+file_name+'.pth')

    if (save_model == True):
        print('Saving model!')
        # torch.save(lidarCNNPoseCov.state_dict(), './model/lidarCNNPoseCov_AzEl_Mini_Pose_Trained_2500.pth')
    # End of Save Model

    # Final Training Results
    print("Final Training Results: loss: %1.10f" % (running_loss/numTrainImg))

# End of Epoch Loop
else: # Training Enable False
    # Load the state dictionary
    lidarCNNPoseCov.load_state_dict(torch.load('./model/lidarCNNPoseCov_AzEl_Mini_Pose_Trained_10000.pth'))
    # lidarCNNPoseCov.load_state_dict(torch.load('./model/lidarCNNPoseCov_Pose_Trained.pth'))
# End of Training

## Fetch the number of double loop files
num_files = trajectory_files.__len__()

cov_size = 9
# Initialize the cycle index
cycle_idx = 0
## Results:
# Initialize the results tensor
# Truth / Vicon
max_traj_idx = numLidarImg
posVehInNavTruth   = torch.zeros(num_files, 2, max_traj_idx)
azVehInNavTruth    = torch.zeros(num_files, 1, max_traj_idx)
covVehInNavTruth   = torch.zeros(num_files, cov_size, max_traj_idx)
# CNN estimate
posVehInNavCNN    = torch.zeros(num_files, 2, max_traj_idx)
azVehInNavCNN     = torch.zeros(num_files, 1, max_traj_idx)
covVehInNavCNN    = torch.zeros(num_files, cov_size, max_traj_idx)

# CNN estimate error
posVehInNavCNNError    = torch.zeros(num_files, 2, max_traj_idx)
azVehInNavCNNError     = torch.zeros(num_files, 1, max_traj_idx)
covVehInNavCNNError    = torch.zeros(num_files, cov_size, max_traj_idx)

max_traj_idx_vec = np.zeros(num_files)

for file_idx in range(num_files):

    # Loading Data
    # Load the Lidar Data Log
    lidarDataLog = sio.loadmat(trajectory_files[file_idx])

    # Pull the 2D lidar image [input]
    lidarCart = lidarDataLog['log']['lidarCart'][0][0]
    lidarCart = torch.tensor(np.float32(lidarCart))

    # Pull the position of the vehicle in the nav frame [target]
    posVehInNav = lidarDataLog['log']['posVehInNav'][0][0]
    posVehInNav = torch.tensor(np.float32(posVehInNav)) / lidarRegFactor

    # Pull the attitude of the vehicle in the nav frame [target]
    azVehInNav = lidarDataLog['log']['azVehInNav'][0][0]
    azVehInNav = torch.tensor(np.float32(azVehInNav))/azRegFactor

    # Pull the covariance of the estimation [target]
    covPoseInNav = lidarDataLog['log']['covariance'][0][0]
    covPoseInNav = torch.tensor(np.float32(covPoseInNav))

    ## Define the training and validation index
    # Index all of the images
    numLidarImg = lidarCart.shape[0]
    lidarImgIdx = np.arange(numLidarImg)

    max_traj_idx_vec[file_idx] = numLidarImg

    print("Loop Number: %d" % (file_idx))

    # Run all of the images through the file
    for img_idx in range(numLidarImg):
        # Pull in the input tensor
        input_tensor = lidarCart[img_idx, :, :].unsqueeze(0)
        # input_tensor = input_tensor.unsqueeze(0)

        # Pull in the target / truth tensors
        pos_target = posVehInNav[:, img_idx]
        att_target = azVehInNav[:, img_idx]
        cov_target = torch.flatten(covPoseInNav[img_idx,:,:])

        # Run the lidar image through the lidar CNN
        pose_pred, cov_pred = lidarCNNPoseCov(input_tensor.to(device))
        pose_pred = Tensor.cpu(pose_pred)
        cov_pred = Tensor.cpu(cov_pred)

        pos_pred = pose_pred[:, 0:2]
        att_pred = pose_pred[:, 2]

        # Convert the covariance prediction into a matrix
        num_train = 1
        cov_mat_pred_L = torch.zeros(num_train, 3, 3)
        cov_mat_pred_D = torch.zeros(num_train, 3, 3)
        cov_mat_pred_L[:, 0, 0] = cov_pred[:, 0]
        cov_mat_pred_L[:, 1, 0] = cov_pred[:, 1]
        cov_mat_pred_L[:, 1, 1] = cov_pred[:, 2]
        cov_mat_pred_L[:, 2, 0] = cov_pred[:, 3]
        cov_mat_pred_L[:, 2, 1] = cov_pred[:, 4]
        cov_mat_pred_L[:, 2, 2] = cov_pred[:, 5]

        cov_mat_pred_D[:, 0, 0] = 1
        cov_mat_pred_D[:, 1, 1] = 1
        cov_mat_pred_D[:, 2, 2] = 1

        cov_mat_pred = torch.zeros(num_train, 3, 3)
        for i in range(num_train):
            cov_mat_pred[i, :, :] = torch.matmul(torch.matmul(cov_mat_pred_L[i, :, :], cov_mat_pred_D[i, :, :]), cov_mat_pred_L[i, :, :].T.squeeze())

        # Log Results Estimates
        posVehInNavTruth[file_idx,:,img_idx] = pos_target * lidarRegFactor
        azVehInNavTruth[file_idx,:,img_idx] = att_target * azRegFactor
        u_idx = [True, False, False, True, True, False, True, True, True]
        covVehInNavTruth[file_idx,:,img_idx] = cov_target

        posVehInNavCNN[file_idx,:,img_idx] = pose_pred.detach()[0, 0:2] * lidarRegFactor
        azVehInNavCNN[file_idx,:,img_idx] = pose_pred.detach()[0][2] * azRegFactor
        covVehInNavCNN[file_idx,:,img_idx] = cov_mat_pred.detach()[0].flatten()

        # Log Results Estimates Error
        posVehInNavCNNError[file_idx,:,img_idx] = (posVehInNavTruth[file_idx,:,img_idx] - posVehInNavCNN[file_idx,:,img_idx])
        azVehInNavCNNError[file_idx,:,img_idx] = (azVehInNavTruth[file_idx,:,img_idx] - azVehInNavCNN[file_idx,:,img_idx])
        covVehInNavCNNError[file_idx,:,img_idx] = (covVehInNavTruth[file_idx,:,img_idx] - covVehInNavCNN[file_idx,:,img_idx])

        del pose_pred, cov_pred

        # Increment the number of cycles
        cycle_idx = cycle_idx + 1
## Results:

traj_idx = np.arange(max_traj_idx)
#----------------------------------------------------------------------------------------------------------------------
# Plot the position estimation history
plt.figure()
for file_idx in range(num_files):
    max_idx = int(max_traj_idx_vec[file_idx])
    plt.plot(posVehInNavTruth[file_idx, 0, 0:max_idx], posVehInNavTruth[file_idx,1, 0:max_idx], 'k.')
    plt.plot(posVehInNavCNN[file_idx, 0, 0:max_idx], posVehInNavCNN[file_idx,1, 0:max_idx], 'b.')
plt.axis('equal')
plt.legend(['Vicon', 'CNN'])
plt.grid()
plt.title('CNN Model 100 Double Loop Position Estimation Performance')

#----------------------------------------------------------------------------------------------------------------------
# Plot the position estimation history
plt.figure()
plt.subplot(2,1,1)
for file_idx in range(num_files):
    max_idx = int(max_traj_idx_vec[file_idx])
    plt.plot(traj_idx[0:max_idx], posVehInNavTruth[file_idx,0,0:max_idx], 'k.')
    plt.plot(traj_idx[0:max_idx], posVehInNavCNN[file_idx,0,0:max_idx], 'b.')
plt.ylabel('Position X Error [mm]')
plt.legend(['Vicon', 'CNN'])
plt.grid()
plt.title('Mean CNN Valid Nav Pos X Error %.4f [mm]' % (torch.mean(posVehInNavCNNError[:,0,:])))

plt.subplot(2,1,2)
for file_idx in range(num_files):
    max_idx = int(max_traj_idx_vec[file_idx])
    plt.plot(traj_idx[0:max_idx], posVehInNavTruth[file_idx,1,0:max_idx], 'k.')
    plt.plot(traj_idx[0:max_idx], posVehInNavCNN[file_idx,1,0:max_idx], 'b.')
plt.ylabel('Position Y Error [mm]')
plt.legend(['Vicon', 'CNN'])
plt.grid()
plt.title('Mean CNN Valid Nav Pos Y Error %.4f [mm]' % (torch.mean(posVehInNavCNNError[:,1,:])))

# Plot the attitude estimation history
plt.figure()
for file_idx in range(num_files):
    max_idx = int(max_traj_idx_vec[file_idx])
    plt.plot(traj_idx[0:max_idx], azVehInNavTruth[file_idx,0,0:max_idx], 'k.')
    plt.plot(traj_idx[0:max_idx], azVehInNavCNN[file_idx,0,0:max_idx], 'b.')
plt.ylabel('Nav Azimuth Angle [deg]')
plt.legend(['CNN', 'CNN-LSTM'])
plt.grid()
plt.suptitle('LSTM Model Attitude Estimation Performance')
plt.title('Mean CNN Valid Nav Az Error %.4f [deg]' % (torch.mean(azVehInNavCNNError)*RAD_TO_DEG))

#----------------------------------------------------------------------------------------------------------------------
# Plot the position estimation error history
plt.figure()
plt.title('CNN-LSTM Model 100 Double Loop Position Estimation Performance')
plt.subplot(2,1,1)
for file_idx in range(num_files):
    max_idx = int(max_traj_idx_vec[file_idx])
    plt.plot(traj_idx[0:max_idx], posVehInNavCNNError[file_idx,0,0:max_idx], 'b.')
plt.ylabel('Position X Error [mm]')
plt.legend(['CNN'])
plt.grid()
plt.title('Mean CNN Valid Nav Pos X Error %.4f [mm]' % (torch.mean(posVehInNavCNNError[:,0,:])))

plt.subplot(2,1,2)
for file_idx in range(num_files):
    max_idx = int(max_traj_idx_vec[file_idx])
    plt.plot(traj_idx[0:max_idx], posVehInNavCNNError[file_idx,1,0:max_idx], 'b.')
plt.ylabel('Position Y Error [mm]')
plt.legend(['CNN'])
plt.grid()
plt.title('Mean CNN Valid Nav Pos Y Error %.4f [mm]' % (torch.mean(posVehInNavCNNError[:,1,:])))

# Plot the attitude estimation error history
plt.figure()
for file_idx in range(num_files):
    max_idx = int(max_traj_idx_vec[file_idx])
    plt.plot(traj_idx[0:max_idx], azVehInNavCNNError[file_idx,0,0:max_idx], 'b.')
plt.ylabel('Nav Azimuth Angle [deg]')
plt.legend(['CNN'])
plt.grid()
plt.suptitle('LSTM Model Attitude Estimation Performance')
plt.title('Mean CNN Valid Nav Az Error %.4f [deg]' % (torch.mean(azVehInNavCNNError)*RAD_TO_DEG))

#----------------------------------------------------------------------------------------------------------------------
# Plot the position estimation error history qq plot
import statsmodels.api as sm

sm.qqplot(posVehInNavCNNError[0:num_files,0,0:max_idx].flatten(), line='s')
plt.title('CNN Nav X Est. Error | Mean: %.4f mm | Std: %.4f mm' % (torch.mean(posVehInNavCNNError[0:num_files,0,0:max_idx]), torch.std(posVehInNavCNNError[0:num_files,0,0:max_idx])))
plt.grid()

sm.qqplot(posVehInNavCNNError[0:num_files,1,0:max_idx].flatten(), line='s')
plt.title('CNN Nav Y Est. Error | Mean: %.4f mm | Std: %.4f mm' % (torch.mean(posVehInNavCNNError[0:num_files,1,0:max_idx]), torch.std(posVehInNavCNNError[0:num_files,1,0:max_idx])))
plt.grid()

sm.qqplot(azVehInNavCNNError[0:num_files,0,0:max_idx].flatten(), line='s')
plt.title('CNN Nav Az Est. Error | Mean: %.4f deg | Std: %.4f deg' % (torch.mean(azVehInNavCNNError[0:num_files,0,0:max_idx]), torch.std(azVehInNavCNNError[0:num_files,0,0:max_idx])))
plt.grid()

plt.figure()
plt.subplot(3,1,1)
plt.hist(posVehInNavCNNError[0:num_files,0,0:max_idx].numpy().flatten(), bins='auto')
plt.title('CNN Nav X Est. Error | Mean: %.4f mm | Std: %.4f mm' % (torch.mean(posVehInNavCNNError[0:num_files,0,0:max_idx]), torch.std(posVehInNavCNNError[0:num_files,0,0:max_idx])))
plt.xlim([-75, 75])
plt.ylabel('Bin Count')
plt.grid()
plt.subplot(3,1,2)
plt.hist(posVehInNavCNNError[0:num_files,1,0:max_idx].numpy().flatten(), bins='auto')
plt.title('CNN Nav Y Est. Error | Mean: %.4f mm | Std: %.4f mm' % (torch.mean(posVehInNavCNNError[0:num_files,1,0:max_idx]), torch.std(posVehInNavCNNError[0:num_files,1,0:max_idx])))
plt.xlim([-75, 75])
plt.ylabel('Bin Count')
plt.grid()
plt.subplot(3,1,3)
plt.hist(azVehInNavCNNError[0:num_files,0,0:max_idx].numpy().flatten() * RAD_TO_DEG, bins='auto')
plt.title('CNN Nav Az Est. Error | Mean: %.4f deg | Std: %.4f deg' % (torch.mean(azVehInNavCNNError[0:num_files,0,0:max_idx]), torch.std(azVehInNavCNNError[0:num_files,0,0:max_idx])))
plt.xlim([-10, 10])
plt.grid()
plt.ylabel('Bin Count')
plt.suptitle('LIDAR CNN 96 Double Loop Histogram')
# Save the workspace variables
workspace_var = { "posVehInNavTruth" : posVehInNavTruth.detach().numpy(), "posVehInNavCNN" : posVehInNavCNN.detach().numpy(),
                  "azVehInNavTruth" : azVehInNavTruth.detach().numpy(), "azVehInNavCNN" : azVehInNavCNN.detach().numpy(),
                  "covVehInNavTruth" : covVehInNavTruth.detach().numpy(), "covVehInNavCNN" : covVehInNavCNN.detach().numpy()}

# savemat("lidarCNN_Cart_Pose_Train_30Lap.mat", workspace_var)
savemat("lidarCNN_Cart_PoseCov_Train_30Lap.mat", workspace_var)

plt.ion()
plt.show(block=True)