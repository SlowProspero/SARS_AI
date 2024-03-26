# Tool for cell and organelle segmentation

## Description
This python-based tool is designed for cell and organelle (nuclei, nucleoli and mitochondria) semantic segmentation, starting from an RI tiff image with a fixed size of 448px\*448px.
The segmentation of whole cells, nuclei and nucleoli are performed with specific U-Net deep learning models, while the mitochondria segmentation is performed by machine learning-based pixel classification.

## Installation
1. Download the project from the GitHub repository (using git or downloading the .zip file).
2. If you don't already have conda installed, download miniconda from https://docs.anaconda.com/free/miniconda/ and follow the installation instructions.
2. Use the terminal to browse to the downloaded project's folder.
3. Create the conda environment by running 'conda env create -n my_env_name -f environment.yml', replacing my_env_name with the desired name for the environment.
Estimated time: less than 5 minutes

Example:
````
cd project_path
conda create -n segmentation_tool_env -f environment.yml
````

## Usage
1. Run 'conda activate my_env_name'
2. Use the terminal to browse to the project's folder
3. Run the following command: 'python main.py my_image_path\my_file_name.tiff', replacing *my_file_path\my_file_name.tiff* with the full path of your image.
4. The script will output the predictions in a 'predictions' folder created where your image is located.
Estimated time: less than 30 seconds

Example:
````
conda activate segmentation_tool_env
cd my_project_path
python main.py example\RI_0001.tiff
````

