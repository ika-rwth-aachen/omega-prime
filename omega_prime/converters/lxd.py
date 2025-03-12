import sys
from pathlib import Path

import betterosi
import numpy as np
from loguru import logger
from lxd_io import Dataset
from tqdm.auto import tqdm
import multiprocessing as mp
from ..asam_odr import MapOdr
from ..recording import Recording


__all__ = ['convert_lxd']
logger.configure(handlers=[{"sink": sys.stdout, "level": "WARNING"}])

NANOS_PER_SEC = 1000000000  # 1 s

vct = betterosi.MovingObjectVehicleClassificationType
vehicles = {
    'Car':  vct.TYPE_CAR,
    'car': vct.TYPE_CAR,
    'Truck': vct.TYPE_HEAVY_TRUCK,
    'truck_bus': vct.TYPE_HEAVY_TRUCK,
    'truck': vct.TYPE_HEAVY_TRUCK,
    'bicycle': vct.TYPE_BICYCLE,
    'van': vct.TYPE_DELIVERY_VAN
}
pedestrians = {
    'pedestrian': betterosi.MovingObjectType.TYPE_PEDESTRIAN

}

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class DatasetConverter:

    def __init__(self, dataset_dir: Path) -> None:
        self._dataset = Dataset(dataset_dir)

    def get_recording_ids(self) -> list[int]:
        return self._dataset.recording_ids

    def get_recording_opendrive_path(self, recording_id: int) -> Path:
        return self._dataset.get_recording(recording_id).opendrive_map_file

    def rec2df(self, recording_id):
        rec = self._dataset.get_recording(recording_id)
        meta = rec._tracks_meta_data
        dt = 1 / rec.get_meta_data("frameRate")
        meta['type'] = meta['class'].apply(lambda x: betterosi.MovingObjectType.TYPE_VEHICLE if x in vehicles else betterosi.MovingObjectType.TYPE_PEDESTRIAN).values
        meta.loc[:,'role'] =  meta['class'].apply(lambda x: betterosi.MovingObjectVehicleClassificationRole.ROLE_CIVIL if x in vehicles else -1)
        meta['subtype'] =  meta['class'].apply(lambda x: vehicles[x] if x in vehicles else -1)
        meta = meta.rename(columns={'trackId': 'idx'})

        tracks = rec._get_tracks_data()
        tracks = tracks.rename(columns={
            'xCenter': 'x',
            'yCenter': 'y',
            'xVelocity': 'vel_x',
            'yVelocity': 'vel_y',
            'xAcceleration': 'acc_x',
            'yAcceleration': 'acc_y',
            'trackId': 'idx',
            'width': 'width',
            'length': 'length'
        })
        for k in ['acc_z', 'z', 'vel_z', 'roll', 'pitch']:
            tracks[k] = 0.0
        tracks['height'] = 2.0
        tracks['yaw'] = wrap_angle(np.deg2rad(tracks["heading"]))
        tracks['total_nanos'] = np.array(tracks['frame']*dt*NANOS_PER_SEC, dtype=int)
        tracks = tracks.merge(meta[['role','idx','type','subtype']], on='idx')
        
        tracks.loc[(tracks['type']==betterosi.MovingObjectType.TYPE_VEHICLE)&(tracks['subtype']==betterosi.MovingObjectVehicleClassificationType.TYPE_BICYCLE), 'width'] = 0.8
        tracks.loc[(tracks['type']==betterosi.MovingObjectType.TYPE_VEHICLE)&(tracks['subtype']==betterosi.MovingObjectVehicleClassificationType.TYPE_BICYCLE), 'length'] = 2
        tracks.loc[(tracks['type']==betterosi.MovingObjectType.TYPE_VEHICLE)&(tracks['subtype']==betterosi.MovingObjectVehicleClassificationType.TYPE_BICYCLE), 'height'] = 1.9
        tracks.loc[(tracks['type']==betterosi.MovingObjectType.TYPE_PEDESTRIAN), 'width'] = 0.5
        tracks.loc[(tracks['type']==betterosi.MovingObjectType.TYPE_PEDESTRIAN), 'length'] = 0.5
        tracks.loc[(tracks['type']==betterosi.MovingObjectType.TYPE_PEDESTRIAN), 'height'] = 1.8
        
        return tracks

def convert_recording(args):
    converter, recording_id, out_filename = args
    tracks = converter.rec2df(recording_id)
    xodr_path = converter.get_recording_opendrive_path(recording_id)
    rec = Recording(df=tracks, map=MapOdr.from_file(xodr_path))
    rec.to_mcap(out_filename)

def convert_lxd(dataset_dir: Path, outpath: Path, n_workers = 1):
    if n_workers == -1:
        n_workers = mp.cpu_count()-1
    outpath = Path(outpath)
    dataset_dir = Path(dataset_dir)
    outpath.mkdir(exist_ok=True)
    converter = DatasetConverter(dataset_dir)
    args_list = [[converter, recording_id, outpath/f'{str(recording_id).zfill(2)}_tracks.mcap'] for recording_id in converter.get_recording_ids()]
    
    if n_workers > 1:
        with mp.Pool(n_workers, maxtasksperchild=1) as pool:
            work_iterator = pool.imap(convert_recording, args_list, chunksize=1)
            list(tqdm(work_iterator, total=len(args_list)))
    else:
        for args in tqdm(args_list):
            convert_recording(args)
