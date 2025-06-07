# Deep Fake Detection

## Repository Content
    .
    ├── DeepFakeDetection             # Deep Fake Detection Python project with FastAPI endpoints 
    ├── FaceRecognitionTest           # C# Console Application for testing
    ├── Test_Results                  # Test results folder
    ├── extract_testdata_CelebDF.py   # Script for Test data preparation
    └── README.md

## Data Preparation

Download the testdata from [here](https://github.com/yuezunli/celeb-deepfakeforensics).

The entire script for preparing the data can be executed locally via a simple command in the terminal on both macOS and Windows.

Make sure the `opencv-python` library is installed using the following command:
  ```
  pip install opencv-python
  ```

  Run the script with the following command:
  ```
  python extract_testdata_CelebDF.py
  ```

## Run project

1. Create virtual environment for python with conda
2. Install libraries from requirements.txt
3. Start `main.py` from the DeepFakeDetection project
4. Select `TestType` in `Program.cs` in FaceRecognitionTest project and specify whether originals or deep fakes must be tested
5. Set API-Keys for the services to be used
6. Start tests

## Results

The test results in the CSV files are stored in the Test_Results folder. 

The Calculation folder contains the following files with the results and for visualizing the ROC curve. Run `analyze_roc` to get ROC curves for specific CSV result files.