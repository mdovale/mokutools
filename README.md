# mokutools

**Liquid Instruments Moku:Lab and Moku:Pro File Management and Data Analysis Tools**

A Python library for file management in Liquid Instruments Moku:Lab and Moku:Pro, with additional features for processing and analyzing of Moku Phasemeter data.

## Features

- **File Management**: List, download, upload, and delete files from Moku devices via network
- **Data Loading**: Load and parse files from Moku Phasemeter measurements
- **Spectral Analysis**: Compute spectral density estimates of phase and frequency via SpecKit
- **Multiple Interfaces**: Use via command-line, Python scripts, or Jupyter notebooks
- **File Format Support**: Handle `.csv`, `.mat`, and `.li` file formats with automatic conversion

## Installation

Install from source:

```bash
git clone https://github.com/mdovale/mokutools.git
cd mokutools
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- ipywidgets >= 7.6.0 (for notebook support)
- requests >= 2.25.0
- py7zr >= 0.20.0
- speckit
- pytdi

## Quick Start

### Loading Phasemeter Data

```python
from mokutools.phasemeter import MokuPhasemeterObject

# Load data from a local file
data = MokuPhasemeterObject('measurement.csv')

# Or download and load directly from a Moku device
data = MokuPhasemeterObject(
    filename='measurement',
    ip='10.128.100.188',
    output_path='./data'
)

# Access the data
print(data.df)  # DataFrame with all channels
print(data.fs)  # Sampling frequency
print(data.nchan)  # Number of channels
```

### File Operations

```python
from mokutools.moku_io import list_files, download, upload, delete

# List files on device
files = list_files('10.128.100.188')

# Download files
download(
    ip='10.128.100.188',
    patterns=['measurement'],
    convert=True,  # Convert .li to .csv
    archive=True,  # Create .zip archive
    output_path='./downloads'
)

# Upload files
upload(ip='10.128.100.188', files=['local_file.csv'])

# Delete files
delete(ip='10.128.100.188', patterns=['temp'])
```

### Interactive CLI

```python
from mokutools.moku_io.cli import (
    download_files_interactive,
    upload_files_interactive,
    delete_files_interactive
)

# Interactive download with prompts
download_files_interactive(
    ip='10.128.100.188',
    file_names=['measurement'],
    convert=True,
    archive=True
)
```

### Jupyter Notebooks

```python
from mokutools.moku_io.notebook import select_file_widget
from mokutools.moku_io import list_files

# Create interactive file selector widget
files = list_files('10.128.100.188')
widget = select_file_widget(files)
display(widget)
```

## Documentation

See the example notebooks in the `notebooks/` directory:

- `0.0_quickstart.ipynb` - Basic usage examples
- `0.2_notebooks.ipynb` - Jupyter notebook integration
- `0.3_phasemeter.ipynb` - Phasemeter analysis examples

## Project Structure

```
mokutools/
├── mokutools/
│   ├── moku_io/          # File I/O operations
│   │   ├── core.py       # Core functions (pure, no I/O)
│   │   ├── cli.py        # Interactive CLI functions
│   │   └── notebook.py   # Jupyter notebook widgets
│   ├── phasemeter.py     # Phasemeter data analysis
│   └── filetools.py      # Backward-compatible wrapper
├── notebooks/            # Example notebooks
├── examples/             # Example scripts
└── tests/                # Test suite
```

## License

BSD 3-Clause License

Copyright (c) 2025, Miguel Dovale

## Author

Miguel Dovale (mdovale@arizona.edu)

## Links

- **Repository**: https://github.com/mdovale/mokutools
- **Issues**: https://github.com/mdovale/mokutools/issues
