# Omega Specification
Data Model, Format and Python Library for ground truth data of dyanmic objects and maps optimized for representing urban traffic.

You might be looking for the now deprecated [OMEGAFormat legacy]() created in VVMethods project.
## Installation
`pip install -r requirements.txt`

## Usage


### Create Omega File from OSI stream and OpenDRIVE

Create Recording form OSI GroundTruth stream and ASAM OpenDrive map and store it to an mcap file.
```python
import omega_format
r = omega_format.Recording.from_file('ground_truths.osi/mcap', xodr_path='map.xodr')
r.to_mcap('test.mcap')

```
### Read and Plot Omega File

```python
import omega_format
r = omega_format.Recording.from_file('test.mcap')
r.plot()

# or 

r.plot_frame(10)
```

### Create Recording from arbitrary object data

You can also directly create a recording from object data in table form (we use pandas). Create a table `df` like the following:

|    |   total_nanos |   idx |       x |        y |   z |     vel_x |     vel_y |   vel_z |   acc_x |   acc_y |   acc_z |   length |   width |   height |   roll |   pitch |      yaw |   type |   subtype |
|---:|--------------:|------:|--------:|---------:|----:|----------:|----------:|--------:|--------:|--------:|--------:|---------:|--------:|---------:|-------:|--------:|---------:|-------:|----------:|
|  0 |             0 |     0 | 131.385 | -34.7839 | 0.9 | -1.29673  | -9.05427  |       0 |       0 |       0 |       0 |  4.03141 | 1.76028 |  1.76028 |      0 |       0 | -1.71305 |      2 |         4 |
|  1 |             0 |     1 | 135.039 | -45.9945 | 0.9 |  0.342834 | -3.44328  |       0 |       0 |       0 |       0 |  3.92792 | 1.84093 |  1.84093 |      0 |       0 | -1.47156 |      2 |         4 |
|  2 |             0 |    12 | 113.346 | -65.7229 | 0.9 | -7.26907  | -4.78079  |       0 |       0 |       0 |       0 |  4.26398 | 1.84183 |  1.84183 |      0 |       0 | -2.55983 |      2 |         4 |
|  3 |             0 |    14 |  89.967 | -73.0296 | 0.9 | -1.1325   | -0.577655 |       0 |       0 |       0 |       0 | 19.0849  | 3.06624 |  3.06624 |      0 |       0 | -2.66992 |      2 |         4 |
|  4 |             0 |    15 | 172.105 | -38.5466 | 0.9 |  0        |  0        |       0 |       0 |       0 |       0 |  4.09254 | 1.82684 |  1.82684 |      0 |       0 | -2.8031  |      2 |         4 |

`type` and `subtyp` have to be integers and correspond to the enumerations `betterosi.MovingObjectType` and `betterosi.MovingObjectVehicleClassificationType`

With such a table `df` you can create a recording:
```python
import pandas as pd
import omega_format
df = pd.DataFrame(...)
omega_format.Recording(df=df)
```
In the next step you would need to add a map.

## File Format
Based on [MCAP](https://mcap.dev/), [ASAM OSI](https://opensimulationinterface.github.io/osi-antora-generator/asamosi/latest/specification/index.html) and [ASAM OpenDRIVE](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html#)
![](omega_specification.svg)
![](docs/omega_format/omega_specification.svg)

In contrast to ASAM OSI the Omega specification sets mandatory singals