# Generated vs Real Maritime Images
The goal of this project is to create a computer vision model that distinguishes between generated and real maritime images. The simulated images are from the SimuShips dataset, and the real images come from the ABOships dataset and the Singapore Maritime Dataset (Visible On-Shore, Visible On-Board, Near-Infrared On-Shore).


## Installation
1. **Clone the repository:**
   ```sh
    git clone https://github.com/vedernikovphoto/Generated-vs-Real-Images.git
    cd Generated-vs-Real-Images
   ```

2. **Install the required packages:**

   If Python 3.10 is already installed:
     ```sh
     make install
     ```
   If you prefer to create a virtual environment:
     ```sh
     make install_venv
     ```
   **⚠️⚠️⚠️ WARNING ⚠️⚠️⚠️**  
   The `make install_venv` command installs CUDA 11.8 and cuDNN 8.  
   This may conflict with existing CUDA/GPU setups. Ensure compatibility before proceeding or edit `setup.sh` to skip CUDA installation.


## Download Datasets
To download and prepare the dataset:

1. **Download the ABOships Dataset:**
    ```sh
    make download_real_data_1
    ```

2. **Download the Singapore Maritime Dataset (Visible On-Shore):**
    ```sh
    make download_real_data_2
    ```

3. **Download the Singapore Maritime Dataset (Visible On-Board):**
    ```sh
    make download_real_data_3
    ```

4. **Download the Singapore Maritime Dataset (Near-Infrared On-Shore):**
    ```sh
    make download_real_data_4
    ```

5. **Download the SimuShips Dataset (Generated):**
    ```sh
    make download_generated_data
    ```

The datasets will be automatically downloaded, preprocessed, and organized in the `data` directory in the project root. Refer to the [Project Report](REPORT.md) for additional details on the preprocessing steps.


## Training the Model
To start training the model:
```sh
make train
```
This command will initiate the training process using the configuration specified in `config/config.yaml`.


## Experiment Tracking
ClearML is used for tracking experiments. Metrics and configurations of the experiments can be viewed directly on ClearML. Access the experiment details [here](https://app.clear.ml/projects/aa3a8726ad1e470380d4b2b654152853/experiments/09f229d0fbce4e1ba18b2023a941df3c/output/execution). 

Ensure that your ClearML credentials are properly configured before running the experiments.


## Model Checkpoints
Model checkpoints will be saved periodically during training in the `experiments` directory. These checkpoints can be used to resume training or for inference. The checkpoint files are named based on the epoch and validation F1 score.


## Running Inference

Place images in the `inference_images` directory and run from the root directory:

```sh
make inference
```

Inference will use the model checkpoint from `model_weights`, saving results to `predictions.csv` in the root directory.

## Linting

The project uses `wemake-python-styleguide` for linting, configured in `src/setup.cfg`. To lint, navigate to `src` and run:

```sh
cd src
flake8 .
```

## Report

For a detailed explanation of the project, including methodology, preprocessing steps, and evaluation metrics, refer to the [Project Report](REPORT.md).

## Repository Structure

```
├── config                        
│   └── config.yaml                 # Configuration files for the training and inference process.             
├── data
│   ├── generated                   # Generated SimuShips dataset.
│   ├── real                        # Real ABOships dataset after preprocessing.
│   ├── real2                       # Real Singapore Maritime Dataset (Visible On-Shore) after preprocessing.
│   ├── real3                       # Real Singapore Maritime Dataset (Visible On-Board) after preprocessing.  
│   └── real4                       # Real Singapore Maritime Dataset (Near-Infrared On-Shore) after preprocessing.
├── experiments                     # Training checkpoints.
├── inference_images                # Input images used during inference.
├── model_weights                   # Stores the best model checkpoint for inference and evaluation.
├── report_images                   # Images for the project report.
├── src                             # Source code for the project.
│   └── preprocessing                    
│       ├── clean_abo_data.py       # Preprocesses the ABOships dataset.
│       └── extract_frames.py       # Extracts frames from videos in the Singapore Maritime Dataset.
│   └── utils                    
│       ├── dataframe_utils.py      # Creates and manages pandas DataFrames.
│       ├── file_utils.py           # Provides file and folder handling functions.
│       ├── logging_utils.py        # Handles functions for detailed logging of dataset splits and metadata.
│       ├── splits_utils.py         # Defines dataset splitting and saving functions.
│       └── train_utils.py          # Includes helper functions for training.
│   ├── augmentations.py            # Defines image augmentations.
│   ├── config.py                   # Manages and loads configurations.
│   ├── constants.py                # Includes project-specific constants.
│   ├── datamodule.py               # Implements the PyTorch Lightning DataModule.
│   ├── dataset_splitter.py         # Splits datasets into train/validation/test sets.
│   ├── dataset.py                  # Handles dataset definition and loading.
│   ├── inference.py                # Performs inference on new images.
│   ├── lightning_module.py         # Implements the PyTorch Lightning module for training.
│   ├── losses.py                   # Defines custom loss functions.
│   ├── metrics.py                  # Implements evaluation metrics.
│   ├── setup.cfg                   # Configuration file for the linter.
│   └── train.py                    # Main script for training the model.
├── .gitignore                      # Specifies files and directories to ignore in version control.
├── venv                            # Virtual environment folder (if created).
├── Makefile                        # Commands and automation for common tasks.
├── predictions.csv                 # Inference results.
├── README.md                       # Project documentation.
├── requirements.txt                # Dependencies for the project.
└── setup.sh                        # Sets up the project virtual environment.
```
