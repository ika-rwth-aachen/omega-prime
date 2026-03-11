"""."""

from omega_prime.metrics.analysis.distance_traveled import distance_traveled
from omega_prime.metrics.analysis.predicted_timegaps import p_timegaps_and_min_p_timegaps
from omega_prime.metrics.analysis.timegaps import timegaps_and_min_timegaps
from omega_prime.metrics.analysis.vel import vel
from omega_prime.metrics.metric_dependencies import append_dependencies, append_if_absent


def test_get_append_if_absent() -> None:
    deps = []
    append_if_absent(vel, deps)
    assert vel in deps
    append_if_absent(vel, deps)
    assert len(deps) == 1


def test_append_dependencies_timegaps() -> None:
    metrics = [timegaps_and_min_timegaps, distance_traveled]
    append_dependencies(metrics)
    assert metrics == [timegaps_and_min_timegaps, distance_traveled]


def test_append_dependencies_predicted_timegaps() -> None:
    metrics = [p_timegaps_and_min_p_timegaps]
    append_dependencies(metrics)
    assert metrics == [p_timegaps_and_min_p_timegaps, vel, distance_traveled, timegaps_and_min_timegaps]
