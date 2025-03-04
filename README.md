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
