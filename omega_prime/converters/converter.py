import sys
from pathlib import Path
from abc import ABC, abstractmethod

from loguru import logger
from tqdm.auto import tqdm
import joblib as jb
from tqdm_joblib import tqdm_joblib
from functools import partial
from ..recording import Recording
from collections.abc import Iterator
from typing import Annotated
import typer


logger.configure(handlers=[{"sink": sys.stdout, "level": "WARNING"}])

NANOS_PER_SEC = 1000000000  # 1 s


class DatasetConverter(ABC):
    def __init__(self, dataset_path: str, out_path: str = "./", n_workers=1) -> None:
        self._dataset_path = Path(dataset_path)
        self._out_path = Path(out_path)
        self.n_workers = n_workers

    @abstractmethod
    def get_source_recordings(self) -> list:
        """
        Abstract method to get a list of the source recordings.
        The method should be implemented in subclasses to handle specific dataset formats.
        Returns:
            source_recordings: List of the source recordings. Could be of any type as further processed in get_recordings.
        """
        pass

    @abstractmethod
    def get_recordings(self, source_recording) -> Iterator:
        """
        Abstract method to get all recordings in a source-recording-instance of the specific dataset.
        The method should be implemented in subclasses to handle specific dataset formats.
        Args:
            source_recordings: List of the source recordings. Could be of any type as returned by get_source_recordings.
        Yields:
            recording: Each recording in the source-recording-instance, one at a time. Could be of any type as further processed in to_omega_prime_recording and get_recording_id.
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
    def get_recording_name(self, recording) -> str:
        """
        Abstract method to get the name for a given recording.
        The method should be implemented in subclasses to handle specific dataset formats.
        Args:
            recording: Recording of any type as returned by get_recordings.
        Returns:
            str: unique name of recording.
        """
        pass

    def convert_source_recording(
        self, source_recording, save_as_parquet: bool = False, skip_existing: bool = False
    ) -> None:
        for recording in self.get_recordings(source_recording):
            out_filename = (
                self._out_path / f"{self.get_recording_name(recording)}.{'parquet' if save_as_parquet else 'mcap'}"
            )
            if not skip_existing or not out_filename.exists():
                Path(out_filename).parent.mkdir(exist_ok=True, parents=True)
                rec = self.to_omega_prime_recording(recording)
                if rec is None:
                    logger.error(f"error during map conversion in the source_recording: {source_recording}")
                else:
                    if save_as_parquet:
                        rec.to_parquet(out_filename)
                    else:
                        rec.to_mcap(out_filename)

    def convert(self, n_workers: int | None = None, save_as_parquet: bool = False, skip_existing: bool = False) -> None:
        if n_workers is None:
            n_workers = self.n_workers
        if n_workers == -1:
            n_workers = jb.cpu_count() - 1
        self._out_path.mkdir(exist_ok=True, parents=True)
        recordings = self.get_source_recordings()
        if n_workers > 1:
            partial_fct = partial(
                self.convert_source_recording, save_as_parquet=save_as_parquet, skip_existing=skip_existing
            )
            with tqdm_joblib(desc="Source Recordings", total=len(recordings)):
                jb.Parallel(n_jobs=n_workers)(jb.delayed(partial_fct)(rec) for rec in recordings)
        else:
            for rec in tqdm(recordings, total=len(recordings)):
                self.convert_source_recording(rec, save_as_parquet=save_as_parquet, skip_existing=skip_existing)

    def yield_recordings(self) -> Iterator[Recording]:
        source_recordings = self.get_source_recordings()
        for sr in tqdm(source_recordings, total=len(source_recordings)):
            for recording in self.get_recordings(sr):
                yield self.to_omega_prime_recording(recording)

    @classmethod
    def convert_cli(
        cls,
        dataset_path: Annotated[
            Path,
            typer.Argument(exists=True, dir_okay=True, file_okay=False, readable=True, help="Root of the dataset"),
        ],
        output_path: Annotated[
            Path,
            typer.Argument(
                file_okay=False, writable=True, help="In which folder to write the created omega-prime files"
            ),
        ],
        n_workers: Annotated[int, typer.Option(help="Set to -1 for n_cpus-1 workers.")] = 1,
        save_as_parquet: Annotated[
            bool,
            typer.Option(
                help="If activated, omega-prime recordings will be stored as parquet files instead of mcap (use for large recordings). Will loose information in OSI that are not mandatory in omega-prime."
            ),
        ] = False,
        skip_existing: Annotated[bool, typer.Option(help="Only convert not yet converted files")] = False,
    ):
        Path(output_path).mkdir(exist_ok=True)
        cls(dataset_path=dataset_path, out_path=output_path, n_workers=n_workers).convert(
            save_as_parquet=save_as_parquet, skip_existing=skip_existing
        )
