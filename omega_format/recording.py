import pandas as pd

from matplotlib import pyplot as plt
import betterosi
import numpy as np
import shapely
import typing
from matplotlib.patches import Polygon as PltPolygon
from .asam_odr import MapOdr
import pandera as pa
from pathlib import Path

pi_valued = pa.Check.between(-np.pi, np.pi)

recording_moving_object_schema = pa.DataFrameSchema(
    columns={
        'x': pa.Column(float), 
        'y': pa.Column(float),
        'z': pa.Column(float),
        'vel_x': pa.Column(float),
        'vel_y': pa.Column(float),
        'vel_z': pa.Column(float),
        'acc_x': pa.Column(float),
        'acc_y': pa.Column(float),
        'acc_z': pa.Column(float),
        'length': pa.Column(float, pa.Check.ge(0)),
        'width': pa.Column(float, pa.Check.ge(0)),
        'height': pa.Column(float),
        'type': pa.Column(int, pa.Check.between(0,4, error=f"Type must be one of {({o.name: o.value for o in betterosi.MovingObjectType})}")),
        'subtype': pa.Column(int,  pa.Check.between(0, 17, error=f"Subtype must be one of {({o.name: o.value for o in betterosi.MovingObjectVehicleClassificationType})}")),
        'roll': pa.Column(float, pi_valued),
        'pitch': pa.Column(float, pi_valued),
        'yaw': pa.Column(float, pi_valued),
        'idx': pa.Column(int, pa.Check.ge(0)),
        'total_nanos': pa.Column(int, pa.Check.ge(0))},
    unique=['idx','total_nanos'])


#recording_internal_mv_schema = recording_moving_object_schema.add_columns({
#    'polygon': pa.Column(shapely.Polygon),
#    'frame': pa.Column(int, pa.Check.ge(0))
#})

def timestamp2ts(timestamp: betterosi
.Timestamp):
    return timestamp.seconds*1_000_000_000+timestamp.nanos

def nearest_interp(xi, x, y):
    # https://stackoverflow.com/a/21003629
    idx = np.abs(x - xi[:,None])
    return y[idx.argmin(axis=1)]
        

class MovingObject():
    def __init__(self, recording, idx):
        super().__init__()
        self.idx = int(idx)
        self._recording = recording
       
        self._df = self._recording._df.loc[self._recording._df['idx']==self.idx]
        self.x = self._df.loc[:,'x'].values
        self.y = self._df.loc[:,'y'].values
        self.z = self._df.loc[:,'z'].values 
        self.vel_x = self._df.loc[:,'vel_x'].values
        self.vel_y = self._df.loc[:, 'vel_y'].values
        self.vel = np.linalg.norm([self.vel_x, self.vel_y], axis=0)
        self.acc_x = self._df.loc[:, 'acc_x'].values
        self.acc_y = self._df.loc[:, 'acc_y'].values
        self.yaw = self._df.loc[:,'yaw'].values
        self.roll = self._df.loc[:, 'roll'].values
        self.pitch = self._df.loc[:, 'pitch'].values
            
        self.timestamps = self._df.loc[:,'total_nanos'].values/float(1e9)
        self.lengths = self._df.loc[:,'length'].values
        self.length = np.mean(self.lengths)
        self.widths = self._df.loc[:,'width'].values
        self.width = np.mean(self.widths)
        self.heights = self._df.loc[:,'height'].values
        self.height = np.mean(self.heights)
        self.polygon = self._df.loc[:,'polygon'].values
        self.type = betterosi.MovingObjectType(self._df.loc[:,'type'].iloc[0])
        self.subtype = betterosi.MovingObjectVehicleClassificationType(self._df.loc[:,'subtype'].iloc[0])
        self.birth = int(self._df.loc[:,'frame'].iloc[0])
        self.end = int(self._df.loc[:,'frame'].iloc[-1])   
    
    def _dfsetter(self, k, val):
        self._recording._df.loc[self._recording._df['idx']==self.idx, k] = val

    @property
    def nanos(self):
        return self.timestamps*1e9

    def plot(self, ax: plt.Axes):
        ax.plot(self.x, self.y, label=str(self.idx), c='red', alpha=.5)
        pass
    
    def plot_mv_frame(self, ax: plt.Axes, frame: int):
        polys = self._df[self._df['frame']==frame]['polygon'].values
        for p in polys:
            ax.add_patch(PltPolygon(p.exterior.coords, fc='red', alpha=0.2))
            
            
class LaneBoundary():
    def __init__(self, osi_lane_boundary, map: "Map"):
        super().__init__()
        self._osi = osi_lane_boundary
        self._map = map
        
        self.idx = self._osi.id.value
        self.polyline = shapely.LineString([(p.position.x, p.position.y) for p in self._osi.boundary_line])
        self.type = betterosi.LaneBoundaryClassificationType(self._osi.classification.type)
    
    def plot(self, ax: plt.Axes):
        ax.plot(*np.array(self.polyline.coords).T, color='gray', alpha=.1)
        
        
                    
class Lane():
    def __init__(self, osi_lane, map: "Map"):
        super().__init__()
        self._osi = osi_lane
        self._map = map
        
        self.idx  = int(self._osi.id.value)
        self._centerline_is_driving_direction = self._osi.classification.centerline_is_driving_direction
        self.centerline = self._get_centerline()
        self.type = betterosi.LaneClassificationType(self._osi.classification.type)
        self.subtype = betterosi.LaneClassificationSubtype(self._osi.classification.subtype)
        self.successor_ids = [p.successor_lane_id.value for p in self._osi.classification.lane_pairing if p.successor_lane_id is not None]
        self.antecessor_ids = [p.antecessor_lane_id.value for p in self._osi.classification.lane_pairing if p.antecessor_lane_id is not None]
        self.right_boundary_ids = [idx.value for idx in self._osi.classification.right_lane_boundary_id if idx is not None]
        self.right_boundary = self._map.lane_boundaries[self.right_boundary_ids[0]]
        self.left_boundary_ids = list([idx.value for idx in self._osi.classification.left_lane_boundary_id if idx is not None])
        self.left_boundary = self._map.lane_boundaries[self.left_boundary_ids[0]]
        self.free_boundary_ids = [idx.value for idx in self._osi.classification.free_lane_boundary_id if idx is not None]
        self.polygon = shapely.Polygon(np.concatenate([np.array(self.left_boundary.polyline.coords), np.flip(np.array(self.right_boundary.polyline.coords), axis=0)]))
        if not self.polygon.is_simple:
            self.polygon = shapely.convex_hull(self.polygon)
            # TODO: fix or warning
            
        # for omega
        self.oriented_borders = self._get_oriented_borders()
        self.start_points = np.array([b.interpolate(0, normalized=True) for b in self.oriented_borders])
        self.end_points = np.array([b.interpolate(1, normalized=True) for b in self.oriented_borders])
    
        
    def _get_centerline(self):
        cl = np.array([(p.x, p.y) for p in self._osi.classification.centerline])
        if not self._centerline_is_driving_direction:
            cl = np.flip(cl, axis=0)
        return cl
    
    def plot(self, ax: plt.Axes):
        c = 'green' if not self.type==betterosi.LaneClassificationType.TYPE_INTERSECTION else 'black'
        ax.plot(*np.array(self.centerline).T, color=c, alpha=0.5)
        ax.add_patch(PltPolygon(self.polygon.exterior.coords, fc='blue', alpha=.2, ec='black'))
        
    # for ase_engine/omega_format

    
    def _get_oriented_borders(self):
        center_start = shapely.LineString(self.centerline).interpolate(0, normalized=True)
        left = self.left_boundary.polyline
        invert_left = left.project(center_start, normalized=True)>.5
        if invert_left:
            left = shapely.reverse(left)
        right = self.right_boundary.polyline
        invert_right = right.project(center_start, normalized=True)>.5
        if invert_right:
            right = shapely.reverse(right)
        return left, right

        


class Map():
    pass

class MapOsi(Map):
    @classmethod
    def from_gt(cls, gt: betterosi.GroundTruth):
        if not hasattr(gt, 'lane') or not hasattr(gt, 'lane_boundary'):
            return None
        else:
            return cls(gt)

    def __init__(self, gt: betterosi.GroundTruth):
        super().__init__()
        self._osi = gt
        
        lane_boundaries = [LaneBoundary(b, self) for b in gt.lane_boundary]
        self.lane_boundaries = {b.idx: b for b in lane_boundaries}
        lanes = [Lane(lane, self) for lane in gt.lane if len(lane.classification.right_lane_boundary_id)>0]
        self.lanes = {l.idx: l for l in lanes}

        
    def plot(self, ax: plt.Axes):
        for l in self.lanes.values():
            l.plot(ax)
        for b in self.lane_boundaries.values():
            b.plot(ax)
            
class Road():
    def __init__(self, parent):
        super().__init__()
        self._parent = parent
        
    @property
    def lanes(self):
        return self._parent.lanes
    


class Recording():
    _MovingObjectClass: typing.ClassVar = MovingObject
    
    @staticmethod
    def _get_polygons(df):
        c2f = df['length']/2
        c2l = df['width']/2
        x = df['x']
        y = df['y']
        cosyaw = np.cos(df['yaw'])
        sinyaw = np.sin(df['yaw']) 
        polys = np.array([
            ((x + (+c2f) * cosyaw- (+c2l) * sinyaw),(y + (+c2f) * sinyaw + (+c2l) * cosyaw)),
            ((x + (+c2f) * cosyaw- (-c2l) * sinyaw),(y + (+c2f) * sinyaw + (-c2l) * cosyaw)),
            ((x + (-c2f) * cosyaw- (-c2l) * sinyaw),(y + (-c2f) * sinyaw + (-c2l) * cosyaw)),
            ((x + (-c2f) * cosyaw- (+c2l) * sinyaw),(y + (-c2f) * sinyaw + (+c2l) * cosyaw))
        ]).swapaxes(0,2).swapaxes(1,2)
        return shapely.polygons(polys)
    
    @staticmethod
    def get_moving_object_ground_truth(nanos: int, df: pd.DataFrame, host_vehicle=None) -> betterosi.GroundTruth:
        recording_moving_object_schema.validate(df)
        mvs = []
        for idx, row in df.iterrows():
            mvs.append(betterosi.MovingObject(
                id=betterosi.Identifier(value=row['idx']),
                type=betterosi.MovingObjectType(row['type']),
                base=betterosi.BaseMoving(
                    dimension=betterosi.Dimension3D(length=row['length'], width=row['width'], height=row['width']),
                    position=betterosi.Vector3D(x=row['x'], y=row['y'], z=row['z']),
                    orientation=betterosi.Orientation3D(roll=row['roll'], pitch=row['pitch'], yaw=row['yaw']),
                    velocity=betterosi.Vector3D(x=row['vel_x'], y=row['vel_y'], z=row['vel_z']),
                    acceleration=betterosi.Vector3D(x=row['acc_x'], y=row['acc_y'], z=row['acc_z']),
                ),
                vehicle_classification=betterosi.MovingObjectVehicleClassification(row['subtype'])
            ))
        gt = betterosi.GroundTruth(
            version=betterosi.InterfaceVersion(version_major=3, version_minor=7, version_patch=9),
            timestamp=betterosi.Timestamp(seconds=int(nanos//1_000_000_000), nanos=int(nanos%1_000_000_000)),
            #host_vehicle_id=betterosi.Identifier(value=0) if host_vehicle is None else betterosi.Identifier(value=host_vehicle),
            moving_object=mvs
        )
        return gt

    def __init__(self, df, map=None, host_vehicle=None, validate=True):
        if validate:
            recording_moving_object_schema.validate(df)
        super().__init__()
        self.nanos2frame = {n: i for i, n in enumerate(df.total_nanos.unique())}
        df['frame'] = df.total_nanos.map(self.nanos2frame)
        if 'polygon' not in df.columns:
            df['polygon'] = self._get_polygons(df)
        self._df = df
        self.map = map
        self.moving_objects = {int(idx): self._MovingObjectClass(self, idx) for idx in self._df['idx'].unique()}
        self.host_vehicle = host_vehicle

    def to_osi_gts(self) -> list[betterosi.GroundTruth]:
        gts = [self.get_moving_object_ground_truth(nanos, group_df, host_vehicle=self.host_vehicle) for nanos, group_df in self._df.groupby('total_nanos')]
        
        if map is not None and isinstance(map, MapOsi):
                gts[0].lane_boundary = [b._osi for b in self.map.lane_boundaries.values()]
                gts[0].lane = [l._osi for l in self.map.lanes.values()]
        return gts
        
    @classmethod
    def from_osi_gts(cls, gts: list[betterosi.GroundTruth]):
        mvs = []
        map = None
        for gt in gts:
            total_nanos = gt.timestamp.seconds*1_000_000_000 + gt.timestamp.nanos
            mvs += [dict(
                total_nanos = total_nanos,
                idx = mv.id.value,
                x = mv.base.position.x,
                y = mv.base.position.y,
                z = mv.base.position.z,
                vel_x = mv.base.velocity.x,
                vel_y = mv.base.velocity.y,
                vel_z = mv.base.velocity.z,
                acc_x = mv.base.acceleration.x,
                acc_y = mv.base.acceleration.y,
                acc_z = mv.base.acceleration.z,
                length = mv.base.dimension.length,
                width = mv.base.dimension.width,
                height = mv.base.dimension.height,
                roll = mv.base.orientation.roll,
                pitch = mv.base.orientation.pitch,
                yaw = mv.base.orientation.yaw,
                type = mv.type,
                subtype = mv.vehicle_classification.type
            ) for mv in gt.moving_object]
            
            if map is None:
                map = MapOsi.from_gt(gt)
        df_mv = pd.DataFrame(mvs).sort_values(by=['total_nanos', 'idx']).reset_index(drop='index')
        return cls(df_mv, map)
    
    @classmethod
    def from_file(cls, filepath):
        gts = betterosi.read(filepath, return_ground_truth=True)
        return cls.from_osi_gts(gts)
    
    @classmethod
    def from_mcap(cls, filepath):
        if Path(filepath).suffix != '.mcap':
            raise ValueError()
        gts = betterosi.read(filepath, return_ground_truth=True)
        r = cls.from_osi_gts(gts)
        try:
            map = MapOdr.from_mcap(filepath)
            r.map = map
        except Exception as e:
            raise e
        return r
    
    def to_mcap(self, filepath):
        if Path(filepath).suffix != '.mcap':
            raise ValueError()
        gts = self.to_osi_gts()
        with betterosi.Writer(filepath) as w:
            for gt in gts:
                w.add(gt)
            if isinstance(self.map, MapOdr):
                w.add(self.map.to_osi(), topic='ground_truth_map', log_time=0)

        
        
    def to_hdf(self, filename):
        #!pip install tables
        self._df.drop(columns=['polygon','frame']).to_hdf(filename, key='moving_object')
        
    @classmethod
    def from_hdf(cls, filename, key='moving_object'):
        df = pd.read_hdf(filename, key=key)
        return cls(df, map=None, host_vehicle=None)
        
    def interpolate(self, new_nanos: list[int]|None = None, hz: float|None = None):
        df = self._df
        if new_nanos is None and hz is None:
            new_nanos = np.linspace(df.total_nanos.min(), df.total_nanos.max(), df.frame.max()-df.frame.min(), dtype=int)
        elif hz is not None:
            step = 1_000_000_000*hz
            new_nanos = np.arange(start=df.total_nanos.min(), stop=df.total_nanos.max()+1, step=step, dtype=int)
        else:
            new_nanos = np.array(new_nanos)
        new_dfs = []
        for idx, track_df in df.groupby('idx'):
            track_data = {}
            track_new_nanos = new_nanos[track_df.frame.min()-df.frame.min():track_df.frame.max()-df.frame.min()+1]
            for c in ['x', 'y', 'z', 'vel_x', 'vel_y', 'vel_z', 'acc_x',
                'acc_y', 'acc_z', 'length', 'width', 'height']:
                track_data[c] = np.interp(track_new_nanos, track_df['total_nanos'], track_df[c])
            for c in ['type', 'subtype']:
                track_data[c] = nearest_interp(track_new_nanos, track_df['total_nanos'].values, track_df[c].values)
            for c in ['roll', 'pitch', 'yaw']:
                track_data[c] = np.mod(np.interp(track_new_nanos, track_df['total_nanos'], np.unwrap(track_df[c], period=np.pi)), np.pi)
            new_track_df = pd.DataFrame(track_data)
            new_track_df['idx'] = idx
            new_track_df['total_nanos'] = track_new_nanos
            new_dfs.append(new_track_df)
        new_df = pd.concat(new_dfs)
        return self.__init__(new_df, self.map, self.host_vehicle)

    def plot(self, ax = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1,1)
        if self.map:
            self.map.plot(ax)
        for ru in self.moving_objects.values():
            ru.plot(ax)
        ax.legend()
        return ax
    
    def plot_frame(self, frame: int, ax = None):
        ax = self.plot(ax=ax)
        self.plot_mv_frame(ax, frame=frame)
        return ax
        
    def plot_mv_frame(self, ax: plt.Axes, frame: int):
        polys = self._df[self._df['frame']==frame]['polygon'].values
        for p in polys:
            ax.add_patch(PltPolygon(p.exterior.coords, fc='red'))
     
