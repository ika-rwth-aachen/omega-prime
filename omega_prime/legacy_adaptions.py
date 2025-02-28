import typing
from datetime import datetime

import betterosi
import numpy as np
import xarray as xr

from .recording import MovingObject, Recording


class Road:
    def __init__(self, parent):
        super().__init__()
        self._parent = parent
        
    @property
    def lanes(self):
        return self._parent.lanes
    
class ParentReroute:
    def __init__(self, parent):
        super().__init__()
        self._parent = parent
        
    def __getattribute__(self, k):
        if k == '_parent':
            return super().__getattribute__(k)
        else:
            return getattr(super().__getattribute__('_parent'), k)
        
        

class MovingObjectOmega(MovingObject):
    
    def __init__(self, recording, idx):
        super().__init__(recording, idx)
        self.tr = ParentReroute(self)
        self.bb = ParentReroute(self)
        self.vel_longitudinal = np.cos(self.yaw)*self.vel_x + np.sin(self.yaw)*self.vel_y
        self.vel_lateral = np.sin(self.yaw)*self.vel_x + np.cos(self.yaw)*self.vel_y
        self.acc_longitudinal = np.cos(self.yaw)*self.acc_x + np.sin(self.yaw)*self.acc_y
        self.acc_lateral = np.sin(self.yaw)*self.acc_x + np.cos(self.yaw)*self.acc_y
        self.vec = np.array([self.length, self.width, self.height])
        self.statistics = {}

    @property
    def is_still(self):
        return np.logical_and(self.vel_lateral<1e-3, self.vel_longitudinal<1e-3)
    @property
    def is_static(self):
        return np.all(self.is_still)
    @property
    def pos_x(self) -> list[float]:
        return self.x
    @property
    def pos_y(self) -> list[float]:
        return self.y
    @property
    def pos_z(self):
        return self.z
    @property
    def heading(self) -> list[float]:
        return self.yaw
    def timespan_to_cutoff_idxs(self, start_idx, end_idx):
        start = self._df[self._df['frame']>=start_idx]['frame'].iloc[0]-self.birth
        end = self._df[self._df['frame']<=end_idx]['frame'].iloc[-1]+1-self.birth
        return start, end, None
    
    @property
    def heading_der(self):
        d = np.diff(self.heading)
        return np.concat([d, [d[0]]])
    
    def in_timespan(self, start_idx, end_idx):
        return len(self._df[np.logical_and(self._df['frame']>=start_idx, self._df['frame']<=end_idx)])>0
    
    def to_xarray(self, _):
        return xr.Dataset(
            {k: ('time', v) for k, v in {'pos_x': self.pos_x, 'pos_y': self.pos_y, 'length': self.length, 'width': self.width, 'heading': self.heading, 'vel_lateral': self.vel_lateral, 'vel_longitudinal': self.vel_longitudinal}.items()},
            coords={'time': self.nanos}
        )
        
        
class RecordingOmega(Recording):
    _MovingObjectClass: typing.ClassVar = MovingObjectOmega
    
    def __init__(self, df, map=None, host_vehicle=None):
        super().__init__(df, map=map, host_vehicle=host_vehicle)
        self.timestamps = np.array(list(self.nanos2frame.keys()))/1e9
        self.daytime = datetime.now()
        self.meta_data = ParentReroute(self)
        self.weather = None
        self.ego_id = None
        if self.map is not None:
            self.roads = {"1": Road(self)}
            self.lanes = self.map.lanes
        else:
            self.lanes = []
            self.roads = {}
        
    @property 
    def dynamic_objects(self):
        return self.moving_objects
    
    @property
    def road_users(self):
        return self.moving_objects
    
    def cut_to_timespan(self, start_idx=None, end_idx=None, inplace=False):
        new_df = self._df
        if start_idx is not None:
            new_df = new_df[new_df['frame']>=start_idx]
        if end_idx is not None:
            new_df = new_df[new_df['frame']<=end_idx]
        if inplace:
            self.__init__(new_df, self.map, self.host_vehicle)
            return self
        else:
            return type(self)(new_df.copy(), map=self.map, host_vehicle=self.host_vehicle)
    
    def get_snippet_tp_ids(self, max_snippets=None, ignore_ego=False):
        if ignore_ego or self.host_vehicle is None:
            ids = [k for k, v in self.road_users.items() if v.type == betterosi.MovingObjectType.TYPE_VEHICLE and v.subtype == betterosi.MovingObjectVehicleClassificationType.TYPE_CAR and not np.all(v.is_still)]
            if max_snippets is not None and max_snippets != 0 and max_snippets != -1:
                ids = ids[:max_snippets]
        else:
            ids = [self.host_vehicle]
        return ids