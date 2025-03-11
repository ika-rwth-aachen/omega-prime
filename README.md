<img src="https://www.ika.rwth-aachen.de/images/ika-logo-a-blau-blau-rgb.svg" align="right" width="240"> 
</br>
</br>

> [!IMPORTANT]  
> The data model and specification are not finalized and are still under discussion. See this repository as a proof of concept.


# Omega-Prime: Data Model, Data Format and Python Library for Handling Ground Truth Traffic Data 
Data Model, Format and Python Library for ground truth data containing information on dynamic objects, map and environmental factors optimized for representing urban traffic. The repository contains:
- **Sepcification Document:** to be released
    - **Data Model**: What signals exists and how these are defined.
    - **Data Format Specification**: How to exchange and store those signals.
- **Python Library**: 
    - **Creation** of omega-prime files from
        - ASAM OSI GroundTruth trace (e.g., output of esmini)
        - Table of moving object data (e.g., csv data)
        - ASAM OpenDRIVE map
    - **Plotting** of data
    - **Validation** of data
    - **Interpolation** of data

The data model and format heavily utilze [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#) and [ASAM Open-Simulation-Interface GroundTruth messages](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/V3.7.0/specification/index.html). omega-prime sets requirements on presence and quality of ASAM OSI GroundTruth messages and ASAM OpenDRIVE files and defines a file format the exchange and storage of such data.

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

## File Format
Based on [MCAP](https://mcap.dev/), [ASAM OSI](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/latest/specification/index.html) and [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#) the ASAM OSI GroundTruth messages and ASAM OpenDRIVE map are packaged as shown in the following figure.
![](https://github.com/ika-rwth-aachen/omega-prime/blob/main/omega_specification.svg)
