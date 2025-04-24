
<img src="https://www.ika.rwth-aachen.de/images/ika-logo-a-blau-blau-rgb.svg" align="right" width="240">
</br>
</br>

[![](https://img.shields.io/badge/license-MPL%202.0-blue.svg)](https://github.com/ika-rwth-aachen/omega-prime/blob/master/LICENSE) 
[![](https://img.shields.io/pypi/v/omega-prime.svg)](https://pypi.python.org/pypi/omega-prime)
[![](https://github.com/ika-rwth-aachen/omega-prime/workflows/CI/badge.svg)](https://github.com/ika-rwth-aachen/omega-prime/actions)
[![](https://img.shields.io/pypi/pyversions/omega-prime.svg)](https://pypi.python.org/pypi/omega-prime/)
[![](https://img.shields.io/github/issues-raw/ika-rwth-aachen/omega-prime.svg)](https://github.com/ika-rwth-aachen/omega-prime/issues)


# Omega-Prime: Data Model, Data Format and Python Library for Handling Ground Truth Traffic Data 

Data Model, Format and Python Library for ground truth data containing information on dynamic objects, map and environmental factors optimized for representing urban traffic. The repository contains:
- **Sepcification Document:** [./docs/omega_prime_specification.md](https://github.com/ika-rwth-aachen/omega-prime/tree/main/docs/omega_prime_specification.md)
    - **Data Model**: What signals exists and how these are defined.
    - **Data Format Specification**: How to exchange and store those signals.
- **Python Library**: 
    - **Creation** of omega-prime files from
        - ASAM OSI GroundTruth trace (e.g., output of esmini)
        - Table of moving object data (e.g., csv data)
        - ASAM OpenDRIVE map
        - [LevelXData datasets](https://levelxdata.com/) through [lxd-io](https://github.com/lenvt/lxd-io)
    - **Plotting** of data
    - **Validation** of data
    - **Interpolation** of data

The data model and format utilze [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#) and [ASAM Open-Simulation-Interface GroundTruth messages](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/V3.7.0/specification/index.html). omega-prime sets requirements on presence and quality of ASAM OSI GroundTruth messages and ASAM OpenDRIVE files and defines a file format for the exchange and storage of these.

Omega-Prime is the successor of the [OMEGAFormat](https://github.com/ika-rwth-aachen/omega_format). It has the benefit that its definition is directly based on the established standards ASAM OSI and ASAM OpenDRIVE and carries over the data quality requirements and the data tooling from OMEGAFormat. Therefore, it should be easier to incorporate omega-prime into existing workflows and tooling. 

To learn more about the example data read [example_files/README.md](https://github.com/ika-rwth-aachen/omega-prime/blob/main/example_files/README.md). Example data was taken and created from [esmini](https://github.com/esmini/esmini).

## Installation
`pip install omega-prime`

## Usage
> A detailed introduction to the features and usage can be found in [tutorial.ipynb](https://github.com/ika-rwth-aachen/omega-prime/blob/main/tutorial.ipynb)

Create an omega-prime file from an OSI GroundTruth message trace and an OpenDRIVE map:
```python
import omega_prime

r = omega_prime.Recording.from_file('example_files/pedestrian.osi', xodr_path='example_files/fabriksgatan.xodr')
r.to_mcap('example.mcap')
```

If you want to create an OSI trace on your own in python, check out the python library [betterosi](https://github.com/ika-rwth-aachen/betterosi).

Read and plot an omega-prime file:

<!--pytest-codeblocks:cont-->
```python
r = omega_prime.Recording.from_file('example.mcap')
ax = r.plot()
```
## Convert Existing Datasets to omega-prime
### [LevelXData](https://levelxdata.com/)
You can convert data from LevelXData to omega-prime. Under the hood [lxd-io](https://github.com/lenvt/lxd-io) is used to perform the conversion.

<!--pytest.mark.skip-->
```python
from omega_prime.converters import LxdConverter
converter = LxdConverter('./exiD-dataset-v2.0', './exiD-as-omega-prime', n_workers=4)
# convert the dataset and store the omega-prime files in the new directory
converter.convert()
# access Recordings directly without storing them
iterator_of_recordings = converter.yield_recordings()
```

or with `omega-prime from-lxd ./exiD-dataset-v2.0 ./exiD-as-omega-prime --n-workers=4`.

Tested with exiD-v2.0 and inD-v1.1

## File Format
Based on [MCAP](https://mcap.dev/), [ASAM OSI](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/latest/specification/index.html) and [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#) the ASAM OSI GroundTruth messages and ASAM OpenDRIVE map are packaged as shown in the following figure.
![](https://github.com/ika-rwth-aachen/omega-prime/blob/main/omega_specification.svg)


# Acknowledgements

This package is developed as part of the [SYNERGIES project](https://synergies-ccam.eu).

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/omega-prime/refs/heads/main/docs/synergies.svg"
style="width:2in" />



Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor the granting authority can be held responsible for them. 

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/omega-prime/refs/heads/main/docs/funded_by_eu.svg"
style="width:4in" />