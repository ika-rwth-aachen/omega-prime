# Creation of example files

The `.osi` files where created with [esmini](https://github.com/esmini/esmini) based on the example files of esmini of the same name found in [resources/xosc and resources/xodr](https://github.com/esmini/esmini/tree/9dc23af113bee6554d0b7606f105be0e252b127b/resources) through the command `.\esmini.exe --window 0 0 800 600 --fixed_timestep 0.033 --osc .\<xosc name>.xosc --osi_file <xosc name>.osi`.

The ASAM OpenDRIVE files are directly taken from the [example files of esmini](https://github.com/esmini/esmini/tree/9dc23af113bee6554d0b7606f105be0e252b127b/resources/xodr).

`mapping.json` stores the `osi` - `xodr` mapping since this is not given from esmini OSI traces.

`example.csv` shows the moving object information in a csv format. It was created by the following script:
```python
import omega_prime
r = omega_format.Recording.from_file('example_files/pedestrian.osi', map_path='example_files/fabriksgatan.xodr')
r._df.drop(columns=['frame','polygon']).to_csv('example_files/example.csv', index=False)
```