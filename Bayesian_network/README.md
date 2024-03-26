# Bayesian network

## Description
This python-based tool establishes the bayesian networks as in figure 6, starting from the single cell quantified dry mass of organelles. 
At its core, the script relies on the bnlearn library.

## Installation
1. Download the project from the GitHub repository (using git or downloading the .zip file).
2. If you don't already have conda installed, download miniconda from https://docs.anaconda.com/free/miniconda/ and follow the installation instructions.
2. Use the terminal to browse to the downloaded project's folder.
3. Create the conda environment by running 'conda env create -n my_env_name -f environment.yml', replacing my_env_name with the desired name for the environment.
Estimated time: less than 5 minutes

Example:
````
cd project_path
conda create -n bayesian_net_env -f environment.yml
````

## Usage
1. Run 'conda activate my_env_name'
2. Use the terminal to browse to the project's folder
3. Run the following command: 'python main.py my_csv_path\my_file_name.csv', replacing *my_csv_path\my_file_name.csv* with the full path of your csv file.
Estimated time: ~ 1 day for 10k iterations

Example:
````
conda activate bayesian_net_env
cd my_project_path
python main.py example\BNdat.csv