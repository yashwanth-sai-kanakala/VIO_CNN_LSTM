# VIO_CNN_LSTM

**Visual Inertial Odometry (VIO)** using Deep Learning models, combining Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) networks for temporal modeling. This method aims to accurately track visual features and handle the temporal dependencies in the odometry estimation. The model is evaluated on the **Euroc MAV Dataset**, which provides real-world data from a quadrotor, including images and inertial measurements.

## Features:
- CNN-based feature extraction from visual input.
- LSTM for temporal modeling to maintain the sequence of movement.
- Integration of visual and inertial data for odometry estimation.

## Dataset:
The **Euroc MAV Dataset** is used for training and testing the model. It contains images, IMU (Inertial Measurement Unit) data, and ground truth data from a real-world MAV (Micro Aerial Vehicle) flight.
- [Euroc MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialodometry)

### Dataset Components:
- **Images**: Provided as RGB frames captured from a camera mounted on the MAV.
- **IMU Data**: Includes measurements such as accelerometer and gyroscope readings.
- **Ground Truth**: The true pose and trajectory of the MAV, used for comparison and evaluation.

## Requirements:

### Dependencies:
The following libraries are required to run this project:

- Python 3.x
- TensorFlow 2.x or PyTorch
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Pandas
- h5py (for data handling)

### Installation:
To install the required dependencies, use:

```bash
pip install -r requirements.txt
```

## Model Overview:

### CNN (Convolutional Neural Network):
- A convolutional neural network (CNN) is used to process the visual input (RGB images).
- It extracts features such as key points, edges, and textures, which are essential for visual odometry.

### LSTM (Long Short-Term Memory):
- LSTM is used to model the temporal relationships between frames.
- It learns the dynamics and trajectory patterns over time from the sequence of images and IMU data.

### Model Architecture:
- The architecture combines CNN for feature extraction and LSTM for sequence learning.
- The CNN layers process each input image, and the LSTM network learns to track the movement over time, enabling the model to predict the current pose or trajectory of the MAV.

## Usage:

1. **Download the Euroc MAV Dataset**:
   - Download the dataset from the official [Euroc MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialodometry).

2. **Data Preprocessing**:
   - Preprocess the images and IMU data into a format compatible with the model.
   - Split the dataset into training and validation sets.

3. **Training**:
   - Train the model using the following command:
     ```bash
     python train.py --dataset_path /path/to/dataset --epochs 100
     ```

4. **Testing**:
   - After training, use the following command to evaluate the model:
     ```bash
     python test.py --model_path /path/to/trained_model --dataset_path /path/to/dataset
     ```

5. **Results**:
   - The model's performance is evaluated by comparing the predicted poses with the ground truth poses provided in the Euroc MAV Dataset.

## Evaluation Metrics:
The model's performance is evaluated using metrics such as:
- **RMSE (Root Mean Squared Error)** between predicted and ground truth poses.
- **Absolute Trajectory Error (ATE)**.
  
## Example:
The following command will run an inference on a test image and print the predicted pose:
```bash
python inference.py --image_path /path/to/image --imu_data /path/to/imu_data --model_path /path/to/trained_model
```

