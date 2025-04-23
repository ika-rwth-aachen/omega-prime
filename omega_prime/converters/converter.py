import sys
from pathlib import Path
from abc import ABC, abstractmethod

from loguru import logger
from tqdm.auto import tqdm
import multiprocessing as mp
from ..recording import Recording

logger.configure(handlers=[{"sink": sys.stdout, "level": "WARNING"}])

NANOS_PER_SEC = 1000000000  # 1 s

class DatasetConverter(ABC):
    def __init__(self, dataset_path: str, out_path: str, n_workers=1) -> None:
        self._dataset_path = Path(dataset_path)
        self._out_path = Path(out_path)
        self.convert(n_workers = n_workers)

    @abstractmethod
    def get_source_recordings(self):
        """
        Abstract method to get a list of the source recordings.
        The method should be implemented in subclasses to handle specific dataset formats.
        Returns:
            source_recordings: List of the source recordings. Could be of any type as further processed in get_recordings.
        """
        pass

    @abstractmethod
    def get_recordings(self, source_recording):
        """
        Abstract method to get all recordings in a source-recording-instance of the specific dataset.
        The method should be implemented in subclasses to handle specific dataset formats.
        Args:
            source_recordings: List of the source recordings. Could be of any type as returned by get_source_recordings.
        Yields:
            recording: Each recording in the dataset, one at a time. Could be of any type as further processed in to_omega_prime_recording and get_recording_id.
        """
        pass

    @abstractmethod
    def to_omega_prime_recording(self, recording) -> Recording:
        """
        Abstract method to convert a raw recording into an omega prime recording instance.
        The method should be implemented in subclasses to handle specific dataset formats.
        Args:
            recording: A recording of any type as returned by get_omega_prime_recordings.
        Returns:
            Recording: An instance of the Recording class containing the processed data.
        """
        pass

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

    def convert_source_recording(self, source_recording) -> None:
        for recording in self.get_recordings(source_recording):
            out_filename = self._out_path / f"{str(self.get_recording_id(recording)).zfill(2)}_tracks.mcap"
            rec = self.to_omega_prime_recording(recording)
            rec.to_mcap(out_filename)

    def convert(self, n_workers=1):
        if n_workers == -1:
            n_workers = mp.cpu_count() - 1
        self._out_path.mkdir(exist_ok=True)
        recordings = self.get_source_recordings()
        if n_workers > 1:
            with mp.Pool(n_workers, maxtasksperchild=1) as pool:
                work_iterator = pool.imap(self.convert_source_recording, recordings, chunksize=1)
                list(tqdm(work_iterator, total=len(recordings)))
        else:
            for rec in tqdm(recordings):
                self.convert_source_recording(rec)