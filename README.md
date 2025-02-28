# Omega-Prime: Data Model, Data Format and Python Library
Data Model, Format and Python Library for ground truth data of dyanmic objects and maps optimized for representing urban traffic. The data model and format heavily utilizes [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#) and [ASAM Open-Simulation-Interface GroundTruth messages](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/V3.7.0/specification/index.html).

To learn more about the example data read [example_files/README.md](example_files/README.md). Example data was taken and created from [esmini](https://github.com/esmini/esmini)


## Features
- Creation
    - ASAM OSI GroundTruth trace (e.g., output of esmini)
    - Dataframe of moving objects (e.g., csv data)
    - ASAM OpenDRIVE map
- Plotting
- Validation
- Interpolatiion
- CLI and Python functions to access features

For a detailed introduction look at [tutorial.ipynb](tutorial.ipynb).

## Installation
`pip install omega-prime`

## File Format
Based on [MCAP](https://mcap.dev/), [ASAM OSI](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/latest/specification/index.html) and [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#)
![](omega_specification.svg)
![](docs/omega_prime/omega_specification.svg)

In contrast to ASAM OSI the Omega specification sets mandatory singals.