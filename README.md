# Brain Hypodensity Segmentation
## Introduction
This project aims to develop a robust and efficient algorithm or AI model for accurately segmenting the hypodense region from Brain Non-Contrast Computed Tomography (NCCT) images. The segmentation process should be invariant to slice thickness and orientation, with the primary goal of automating and streamlining the identification of early ischemic changes in acute stroke patients.

## Features
- **Hypodensity Segmentation:** Accurate identification and segmentation of hypodense regions in brain NCCT images.
- **Slice Thickness and Orientation Invariance:** The model should perform well regardless of variations in slice thickness and orientation.
- **Automation for Stroke Identification:** The algorithm aims to automate the identification of early ischemic changes, contributing to the efficient diagnosis of acute stroke patients.

## Getting Started

### Prerequisites

- Python 3
- TensorFlow
- NumPy
- nibabel
-matplotlib
### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/brain-hypodensity-segmentation.git

### Prerequisites

Make sure you have the following prerequisites installed before running the code:

- [Python 3](https://www.python.org/downloads/)
- [TensorFlow](https://www.tensorflow.org/install)
- [NumPy](https://numpy.org/install/)
- [nibabel](https://nipy.org/nibabel/)
- [Matplotlib](https://matplotlib.org/stable/users/installing.html)

```bash
pip install tensorflow numpy nibabel matplotlib
### Data Setup

The dataset for this project should adhere to the following structure:

```plaintext
dataset/
|-- CaseID1/
|   |-- CaseID1_NCCT.nii.gz
|   |-- CaseID1_ROI.nii.gz
|-- CaseID2/
|   |-- CaseID2_NCCT.nii.gz
|   |-- CaseID2_ROI.nii.gz
...
# Set your data path
data_path = '/path/to/your/dataset/'

Use the provided BrainHypoDataGenerator class to generate batches of training and validation data.
# Example usage
from BrainHypoDataGenerator import BrainHypoDataGenerator

# Set your data path
data_path = '/path/to/your/dataset/'

# Get the list of train images and masks
train_images = sorted(glob(os.path.join(data_path, '*', '*_NCCT.nii.gz')))
train_masks = sorted(glob(os.path.join(data_path, '*', '*_ROI.nii.gz')))

# Set batch size and image size
batch_size = 1
image_size = (512, 512)

# Create data generators
train_generator = BrainHypoDataGenerator(train_images[:10], train_masks[:10], batch_size, image_size)
val_generator = BrainHypoDataGenerator(train_images[10:], train_masks[10:], batch_size, image_size)









