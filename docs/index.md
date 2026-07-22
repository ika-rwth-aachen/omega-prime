# Omega-Prime: Data Model, Data Format and Python Library for Handling Ground Truth Road Traffic Data 

Data Model, Format and Python Library for ground truth road traffic data containing information on dynamic objects, map and environmental factors optimized for representing urban traffic. The repository contains:
## Data Model and Specification
see [Data Model & Specification](omega_prime_specification.md)

- 🌍 **Data Model**: What signals exist and how these are defined.
- 🧾 **Data Format Specification**: How to exchange and store those signals.

## Python Library
  - 🔨 **Create** omega-prime files from many sources (see [Tutorials/Introduction](notebooks/tutorial.ipynb)):
      - ASAM OSI GroundTruth trace (e.g., output of esmini),  Table of moving object data (e.g., csv data), ASAM OpenDRIVE map
      - [LevelXData datasets](https://levelxdata.com/) through [lxd-io](https://github.com/lenvt/lxd-io)
      - Extend yourself by subclassing [DatasetConverter](https://github.com/ika-rwth-aachen/omega-prime/blob/main/omega_prime/converters/converter.py#L52)
      - Use [omega-prime-trajdata](https://github.com/ika-rwth-aachen/omega-prime-trajdata) to convert motion prediction datasets into omega-prime 
  - 🗺️ **Map Association**: Associate Object Location with Lanes from OpenDRIVE or OSI Maps (see [Tutorials/Locator](notebooks/tutorial_locator.ipynb))
  - 📺 **Plotting** of data: interactive top view plots using [altair](https://altair-viz.github.io/)
  - ✅ **Validation** of data: check if your data conforms to the omega-prime specification (e.g., correct yaw) using [pandera](https://pandera.readthedocs.io/en/stable/)
  - 📐 **Interpolation** of data: bring your data into a fixed frequency
  - 📈 **Metrics**: compute interaction metrics like PET, TTC, THW (see [Tutorials/Metrics](notebooks/tutorial_metrics.ipynb))
    - Predicted and observed timegaps based on driving tubes (see [./omega_prime/metrics/analysis](https://github.com/ika-rwth-aachen/omega-prime/blob/main/omega_prime/metrics/analysis))
    - 2D-birds-eye-view visibility with [omega-prime-visibility](https://github.com/ika-rwth-aachen/omega-prime-visibility)
  - 🚀 **Fast Processing** directly on DataFrames using [polars](https://pola.rs/), [polars-st](https://oreilles.github.io/polars-st/)
  - ⌨️ **CLI** to convert, validate and visualize omega-prime files

### ROS 2 Conversion
  - Tooling for conversion from ROS 2 bag-files containing [perception_msgs::ObjectList](https://github.com/ika-rwth-aachen/perception_interfaces/blob/main/perception_msgs/msg/ObjectList.msg) messages to omega-prime MCAP is available in `tools/ros2_conversion/`.
  - A Dockerfile and  [usage instructions](tools/ros2_conversion/README.md) are provided explaining how to run the export end-to-end.

## Installation
`pip install omega-prime`


## Acknowledgements

This package is developed as part of the [SYNERGIES project](https://synergies-ccam.eu).

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/omega-prime/refs/heads/main/docs/synergies.svg"
style="width:2in" />



Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Climate, Infrastructure and Environment Executive Agency (CINEA). Neither the European Union nor the granting authority can be held responsible for them. 

<img src="https://raw.githubusercontent.com/ika-rwth-aachen/omega-prime/refs/heads/main/docs/funded_by_eu.svg"
style="width:4in" />

## Notice
> The project is open-sourced and maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/).
> We cover a wide variety of research topics within our [*Vehicle Intelligence & Automated Driving*](https://www.ika.rwth-aachen.de/en/competences/fields-of-research/vehicle-intelligence-automated-driving.html) domain.
> If you would like to learn more about how we can support your automated driving or robotics efforts, feel free to reach out to us!
> ***opensource@ika.rwth-aachen.de***
