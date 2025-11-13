
<img src="https://github.com/ika-rwth-aachen/omega-prime/blob/main/docs/logo/omega-prime.svg?raw=True" height=150px align="right" style="margin: 10px;">

[![](https://img.shields.io/badge/license-MPL%202.0-blue.svg)](https://github.com/ika-rwth-aachen/omega-prime/blob/master/LICENSE) 
[![](https://img.shields.io/pypi/v/omega-prime.svg)](https://pypi.python.org/pypi/omega-prime)
[![](https://github.com/ika-rwth-aachen/omega-prime/workflows/CI/badge.svg)](https://github.com/ika-rwth-aachen/omega-prime/actions)
[![](https://img.shields.io/pypi/pyversions/omega-prime.svg)](https://pypi.python.org/pypi/omega-prime/)
[![](https://img.shields.io/github/issues-raw/ika-rwth-aachen/omega-prime.svg)](https://github.com/ika-rwth-aachen/omega-prime/issues)
[![](https://img.shields.io/badge/Documentation-2e8b57)](https://ika-rwth-aachen.github.io/omega-prime)

# Omega-Prime: Data Model, Data Format and Python Library for Handling Ground Truth Traffic Data 

Data Model, Format and Python Library for ground truth data containing information on dynamic objects, map and environmental factors optimized for representing urban traffic. The repository contains:
### Data Model and Specification
see [Data Model & Specification](https://ika-rwth-aachen.github.io/omega-prime/omega_prime_specification/)

- 🌍 **Data Model**: What signals exist and how these are defined.
- 🧾 **Data Format Specification**: How to exchange and store those signals.

### Python Library
  - 🔨 **Create** omega-prime files from many sources (see [./docs/notebooks/tutorial.ipynb](https://github.com/ika-rwth-aachen/omega-prime/blob/main/docs/notebooks/tutorial.ipynb)):
      - ASAM OSI GroundTruth trace (e.g., output of esmini)
      - Table of moving object data (e.g., csv data)
      - ASAM OpenDRIVE map
      - [LevelXData datasets](https://levelxdata.com/) through [lxd-io](https://github.com/lenvt/lxd-io)
      - Extend yourself by subclassing [DatasetConverter](omega_prime/converters/converter.py)
      - Use [omega-prime-trajdata](https://github.com/ika-rwth-aachen/omega-prime-trajdata) to convert motion prediction datasets into omega-prime 
  - 🗺️ **Map Association**: Associate Object Location with Lanes from OpenDRIVE or OSI Maps (see [./docs/notebooks/tutorial_locator.ipynb](https://github.com/ika-rwth-aachen/omega-prime/tree/main/docs/notebooks/tutorial_locatory.ipynb))
  - 📺 **Plotting** of data: interactive top view plots using [altair](https://altair-viz.github.io/)
  - ✅ **Validation** of data: check if your data conforms to the omega-prime specification (e.g., correct yaw) using [pandera](https://pandera.readthedocs.io/en/stable/)
  - 📐 **Interpolation** of data: bring your data into a fixed frequency
  - 📈 **Metrics**: compute interaction metrics like PET, TTC, THW (see [./docs/notebooks/tutorial_metrics.ipynb](https://github.com/ika-rwth-aachen/omega-prime/tree/main/docs/notebooks/tutorial_metrics.ipynb))
    - Predicted and observed timegaps based on driving tubes (see [./omega_prime/metrics.py](https://github.com/ika-rwth-aachen/omega-prime/blob/main/omega_prime/metrics.py))
    - 2D-birds-eye-view visibility with [omega-prime-visibility](https://github.com/ika-rwth-aachen/omega-prime-visibility)
  - 🚀 **Fast Processing** directly on DataFrames using [polars](https://pola.rs/), [polars-st](https://oreilles.github.io/polars-st/)

### ROS 2 Conversion
  - Tooling for conversion from ROS 2 bag-files containing [perception_msgs::ObjectList](https://github.com/ika-rwth-aachen/perception_interfaces/blob/main/perception_msgs/msg/ObjectList.msg) messages to omega-prime MCAP is available in `tools/ros2_conversion/`.
  - A Dockerfile and  [usage instructions](tools/ros2_conversion/README.md) are provided explaining how to run the export end-to-end.

The data model and format utilize [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#) and [ASAM Open-Simulation-Interface GroundTruth messages](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/V3.7.0/specification/index.html). omega-prime sets requirements on presence and quality of ASAM OSI GroundTruth messages and ASAM OpenDRIVE files and defines a file format for the exchange and storage of these.

Omega-Prime is the successor of the [OMEGAFormat](https://github.com/ika-rwth-aachen/omega_format). It has the benefit that its definition is directly based on the established standards ASAM OSI and ASAM OpenDRIVE and carries over the data quality requirements and the data tooling from OMEGAFormat. Therefore, it should be easier to incorporate omega-prime into existing workflows and tooling. 

To learn more about the example data read [example_files/README.md](https://github.com/ika-rwth-aachen/omega-prime/blob/main/example_files/README.md). Example data was taken and created from [esmini](https://github.com/esmini/esmini).

## Installation
`pip install omega-prime`

## Usage
> A detailed introduction to the features and usage can be found in [./docs/notebooks/tutorial.ipynb](https://github.com/ika-rwth-aachen/omega-prime/blob/main/docs/notebooks/tutorial.ipynb)

Create an omega-prime file from an OSI GroundTruth message trace and an OpenDRIVE map:
```python
import omega_prime

r = omega_prime.Recording.from_file('example_files/pedestrian.osi', map_path='example_files/fabriksgatan.xodr')
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

Tested with exiD-v2.0, inD-v1.1, highD-v1.0 (highD does not provide an ASAM OpenDRIVE map).

## File Format
Based on [MCAP](https://mcap.dev/), [ASAM OSI](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/latest/specification/index.html) and [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#) the ASAM OSI GroundTruth messages and ASAM OpenDRIVE map are packaged as shown in the following figure.
![](https://github.com/ika-rwth-aachen/omega-prime/blob/main/docs/omega_prime/omega_specification.svg)


# Acknowledgements

This package is developed as part of the [SYNERGIES project](https://synergies-ccam.eu).

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/omega-prime/refs/heads/main/docs/synergies.svg"
style="width:2in" />



Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor the granting authority can be held responsible for them. 

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/omega-prime/refs/heads/main/docs/funded_by_eu.svg"
style="width:4in" />

# Notice

> [!IMPORTANT]
> The project is open-sourced and maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/).
> We cover a wide variety of research topics within our [*Vehicle Intelligence & Automated Driving*](https://www.ika.rwth-aachen.de/en/competences/fields-of-research/vehicle-intelligence-automated-driving.html) domain.
> If you would like to learn more about how we can support your automated driving or robotics efforts, feel free to reach out to us!
> :email: ***opensource@ika.rwth-aachen.de***
