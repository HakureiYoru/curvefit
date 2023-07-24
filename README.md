# curvefit

# Project Title

## Overview

This project consists of a data analysis and fitting application written in Python. The scripts utilize a set of parametric equations for fitting and analyzing calibration data, and subsequently applying the results to test data, which are then visualized in a map.

## Scripts

The project consists of three main Python scripts:

1. `UI.py`: This script provides a graphical user interface (GUI) that allows users to load data, set parameter boundaries, initiate a fitting process, view logs, and choose whether to autoscale graphs.

2. `Filter.py`: This script defines a set of parametric equations and uses these to fit the provided x and y data. The fitting process is performed in a separate thread to avoid blocking the GUI. The results of the fitting process are logged and saved to a file.

3. `function_analysis.py`: This script contains several functions used for further processing and analysis of the x and y data. Functions include Fourier transformations, keeping data within one period, and generating maps of the data.

## Workflow

1. **Calibration**: Load the calibration data (end0). Upon loading, a window displaying the pre-read parameter results will pop up.

2. **Fitting**: Click the "Fit" button to execute the fitting process. The result is a set of calibrated reference data.

3. **Testing**: After fitting, open a new "map" window. In this window, load the actual test data. The results of the test data will be displayed on various maps.

## Requirements

- Python 3.7 or later
- Libraries: numpy, scipy, Tkinter, json, logging

## Installation

1. Clone this repository to your local machine.
2. Install the necessary libraries by running `pip install -r requirements.txt` in your terminal.

## Usage

1. Run the `UI.py` script to launch the graphical user interface.
2. Load your data file, set parameter boundaries as needed.
3. Click "Fit" to start the fitting process.
4. After fitting, open the "map" window and load the test data.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License


