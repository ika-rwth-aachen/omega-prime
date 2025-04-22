import sys
from pathlib import Path
from abc import ABC, abstractmethod

from loguru import logger
from tqdm.auto import tqdm
import multiprocessing as mp
from ..map_odr import MapOdr
from ..recording import Recording
import polars as pl

logger.configure(handlers=[{"sink": sys.stdout, "level": "WARNING"}])

NANOS_PER_SEC = 1000000000  # 1 s

class DatasetConverter(ABC):
    def __init__(self, dataset_path: str, out_path: str, n_workers=1) -> None:
        self._dataset_path = Path(dataset_path)
        self._out_path = Path(out_path)
        self.convert(n_workers = n_workers)

    @abstractmethod
    def get_recordings(self):
        """
        Abstract method to get all recording paths in the dataset.
        The method should be implemented in subclasses to handle specific dataset formats.
        Returns:
            List of recordings. Could be of any type as further processed in rec2df, get_recording_opendrive_path and get_recording_id.
        """
        pass

    @abstractmethod
    def rec2df(self, recording) -> pl.DataFrame:
        """
        Abstract method to load raw data from the recording path.
        The method should be implemented in subclasses to handle specific dataset formats.
        Args:
            recording: Recording of any type as returned by get_recordings.
        Returns:
            pl.DataFrame: DataFrame containing the processed data as specified in TODO.
        """
        pass

    @abstractmethod
    def get_recording_opendrive_path(self, recording) -> Path:
        """
        Abstract method to get the OpenDRIVE path for a given recording.
        The method should be implemented in subclasses to handle specific dataset formats.
        Args:
            recording: Recording of any type as returned by get_recordings.
        Returns:
            Path: Path to the OpenDRIVE file associated with the recording. Returns None if not available.
        """
        return None

    @abstractmethod
    def get_recording_id(self, recording) -> int:
        """
        Abstract method to get the recording ID for a given recording.
        The method should be implemented in subclasses to handle specific dataset formats.
        Args:
            recording: Recording of any type as returned by get_recordings.
        Returns:
            int: Recording ID.
        """
        pass

    def convert_source_recording(self, recording) -> None:
        out_filename = self._out_path / f"{str(self.get_recording_id(recording)).zfill(2)}_tracks.mcap"
        tracks = self.rec2df(recording)
        xodr_path = self.get_recording_opendrive_path(recording)
        rec = Recording(df=tracks, map=MapOdr.from_file(xodr_path), validate=False)
        rec.to_mcap(out_filename)

    def convert(self, n_workers=1):
        if n_workers == -1:
            n_workers = mp.cpu_count() - 1
        self._out_path.mkdir(exist_ok=True)
        recordings = self.get_recordings()
        if n_workers > 1:
            with mp.Pool(n_workers, maxtasksperchild=1) as pool:
                work_iterator = pool.imap(self.convert_source_recording, recordings, chunksize=1)
                list(tqdm(work_iterator, total=len(recordings)))
        else:
            for rec in tqdm(recordings):
                self.convert_source_recording(rec)