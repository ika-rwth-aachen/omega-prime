from .analysis.distance_traveled import distance_traveled
from .analysis.predicted_timegaps import p_timegaps_and_min_p_timegaps
from .analysis.timegaps import timegaps_and_min_timegaps
from .analysis.vel import vel
from .metric import Metric


def append_if_absent(metric: Metric, metrics: list[Metric]) -> None:
    if metric not in metrics:
        metrics.append(metric)


def append_dependencies(metrics: list[Metric]) -> None:
    if timegaps_and_min_timegaps in metrics:
        append_if_absent(distance_traveled, metrics)

    if p_timegaps_and_min_p_timegaps in metrics:
        append_if_absent(vel, metrics)
        append_if_absent(distance_traveled, metrics)
        append_if_absent(timegaps_and_min_timegaps, metrics)
