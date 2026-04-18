# Urban Sound Classification for IOT Sensors

This project classifies environmental audio clips from the UrbanSound8K dataset into 10 distinct categories using Convolutional Neural Networks (CNNs) . The repository explores various audio feature extraction techniques, including MFCCs, Log-Mel Spectrograms, and a combined feature approach, alongside model interpretability (Grad-CAM) and latency optimization (8-bit Quantization).

## Dataset

This project requires the **UrbanSound8K** dataset.
* **Official Website & Download:** [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html)
* **Kaggle Link:** [UrbanSound8K on Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

**Important:** Download the dataset and extract it directly into the root of this project so that the path looks exactly like this:
```text
SPEECH_PROJECT/
├── UrbanSound8K/
│   └── UrbanSound8K/
│       ├── metadata/
│       │   └── UrbanSound8K.csv
│       └── audio/
│           ├── fold1/
│           ├── ...
│           └── fold10/

1. Installation & Setup
      Clone or download this repository.

2.Create a virtual environment (recommended):
      python -m venv env
      source env/bin/activate  # On Windows use: env\Scripts\activate

3.Install the dependencies:
      pip install -r requirements.txt


Project Structure
The project is divided into four main modeling approaches, each housed in its own directory:

1.model1/: Contains the baseline CNN architecture (m1.py) where feature extraction is done using mfccs.
2.model2/: Uses 128-band Log-Mel Spectrograms for feature extraction (m2.py), optimizing for how human hearing perceives sound frequencies.
3.model3/: Stacks MFCCs, Mel Spectrograms, and Spectral Contrast into a rich 175-band feature map (m3.py) for maximum accuracy.
4.optimised_model/: Uses 40-band MFCCs as input features (om.py). Includes 8-bit Post-Training Quantization (PTQ) and Grad-CAM visualizations.

🚀 How to Run the Models
Since each model relies on the dataset being in the root directory, you should run the scripts from the root SPEECH_PROJECT directory, or ensure the internal paths in the scripts point to the correct dataset location.

Running the Model-1
Bash
cd model1
python m1.py
Outputs (Plots, text summaries, and .pt weights) will be saved in the /results1/ folder.

Running the Model-2
Bash
cd model2
python m2.py
Outputs will be saved in the /results_melspec/ folder.

Running the  Model-3
Bash
cd model3
python m3.py
Outputs will be saved in the results_combined_features/ folder.

Running the Optimised Model
Bash
cd optimised_model
python om.py
Outputs will be saved in the results/ folder.




