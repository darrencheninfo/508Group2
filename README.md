# 508Group2: Enhancing Breast Cancer Detection Through AI

**ADS508 Data Science With Cloud Computing Project – Group 2**  
**Authors:** Arjun Venkatesh, Darren Chen, Vinh Dao  
**University of San Diego, MS-ADS, ADS‑508**

---

## Overview

Breast cancer remains one of the leading causes of mortality among women worldwide. Early detection significantly improves survival rates, yet existing screening methods often rely heavily on manual analysis—which can be time-consuming, error-prone, and inaccessible in some regions.

To address this challenge, our project leverages AI-driven models (specifically XGBoost and deep learning techniques) to assist radiologists in detecting breast cancer at an early stage with high accuracy. Our solution focuses on:

- **Data Security & Cloud Deployment:** Storing and processing data in AWS S3 and SageMaker, ensuring compliance with healthcare privacy regulations.
- **Robust Training Pipeline:** Implementing sophisticated data preprocessing, feature engineering, and model training routines.
- **Explainable & Ethical AI:** Emphasizing transparency to support radiologists rather than replace them, and auditing for potential biases.

---

## Repository Contents

Below is the structure of the repository:

    508Group2/
    ├── README.md                # Project overview and instructions (this file)
    ├── LICENSE                  # License information (optional)
    ├── .gitignore               # Files and directories to be ignored by Git
    ├── environment.yml          # Conda environment file (or use requirements.txt)
    ├── docs/                    # Documentation & design materials
    │   ├── ADS508_Design_Document.pdf
    │   └── Presentation.pdf
    ├── src/                     # Main source code
    │   ├── __init__.py
    │   ├── data_processing.py   # Data ingestion & preprocessing
    │   ├── model_training.py    # Model definition & training routines
    │   └── utils.py             # Helper functions & utilities
    ├── notebooks/               # Jupyter Notebooks for EDA & prototyping
    │   └── exploratory_analysis.ipynb
    ├── data/                    # (Optional) raw/processed data files
    │   ├── raw/
    │   └── processed/
    └── results/                 # Outputs, model artifacts, logs, etc.

**Notes:**
- The **docs/** folder contains the primary design document and any presentation slides.
- The **src/** folder contains Python scripts for data processing, model training, etc.
- The **notebooks/** folder contains exploratory notebooks to visualize and experiment with data.
- The **data/** folder holds raw or processed datasets (if version controlling large files, consider using `.gitignore` or an external storage solution).
- The **results/** folder can store trained models, evaluation metrics, or logs.

---

## Cloning or Pulling the Repository

### Option 1: Cloning the Repository

1. **Open a terminal** in your local environment or within your JupyterLab workspace.  
2. **Clone the repository** using: `git clone https://github.com/darrencheninfo/508Group2.git`  
3. **Navigate** into the project folder: `cd 508Group2`

### Option 2: Pulling Updates (if you already have the repo)

1. **Open a terminal** in your existing local clone or JupyterLab workspace.  
2. **Pull the latest changes**: `git pull origin main`

---

## Installation & Setup

### Prerequisites
- Python 3.7+  
- Conda (recommended) or pip

### Conda Environment (Recommended)

1. **Navigate** to the project root: `cd 508Group2`  
2. **Create the environment**: `conda env create -f environment.yml`  
3. **Activate the environment**: `conda activate 508group2-env`

> Alternatively, if you use `requirements.txt`, run: `pip install -r requirements.txt`

### AWS Credentials (Optional)
If the workflow involves AWS S3 or SageMaker, ensure your AWS credentials are properly configured: `aws configure`

---

## Usage

### Running Model Training
Within the project root: `python src/model_training.py`  
- This script handles data ingestion, preprocessing, model training, and outputs performance metrics.

### Exploratory Analysis
Launch Jupyter Notebook for data exploration: `jupyter notebook notebooks/exploratory_analysis.ipynb`  
- Here, you can view visualizations, test different modeling approaches, and fine-tune parameters interactively.

### Documentation
- Additional details are in the **docs/** folder:
  - `ADS508_Design_Document.pdf`
  - `Presentation.pdf`

---

## Contributing

1. **Fork** the repository on GitHub.  
2. **Create a new branch** for your feature/fix: `git checkout -b feature/your_feature`  
3. **Commit** your changes and **push** to your fork:
4. **Open a Pull Request** on GitHub describing the changes made.

---

## Contact

For questions or further collaboration, please reach out to:
- **Arjun Venkatesh**
- **Darren Chen**
- **Vinh Dao**

University of San Diego (MS-ADS, ADS‑508)

---

*Thank you for using our project! We hope this AI-driven solution accelerates the fight against breast cancer.*
