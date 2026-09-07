import pytest

from load_balancer.round_robin import RoundRobinLoadBalancer
from workers.worker import MLWorker


def test_round_robin_cycles_through_workers():
    workers = [
        MLWorker("worker-1"),
        MLWorker("worker-2"),
        MLWorker("worker-3"),
    ]

    load_balancer = RoundRobinLoadBalancer(workers)

    assert load_balancer.get_next_worker().worker_id == "worker-1"
    assert load_balancer.get_next_worker().worker_id == "worker-2"
    assert load_balancer.get_next_worker().worker_id == "worker-3"
    assert load_balancer.get_next_worker().worker_id == "worker-1"


def test_round_robin_skips_unhealthy_workers():
    workers = [
        MLWorker("worker-1"),
        MLWorker("worker-2"),
        MLWorker("worker-3"),
    ]

    workers[1].is_healthy = False

    load_balancer = RoundRobinLoadBalancer(workers)

    assert load_balancer.get_next_worker().worker_id == "worker-1"
    assert load_balancer.get_next_worker().worker_id == "worker-3"
    assert load_balancer.get_next_worker().worker_id == "worker-1"
    assert load_balancer.get_next_worker().worker_id == "worker-3"

def test_raises_error_when_no_workers_are_healthy():
    workers = [
        MLWorker("worker-1"),
        MLWorker("worker-2"),
        MLWorker("worker-3"),
    ]
    for worker in workers:
        worker.is_healthy=False

    load_balancer=RoundRobinLoadBalancer(workers)

    with pytest.raises(RuntimeError, match="No healthy workers available"):
        load_balancer.get_next_worker()

def test_unhealthy_worker_can_recover():
    workers = [
        MLWorker("worker-1"),
        MLWorker("worker-2"),
        MLWorker("worker-3"),
    ]

    workers[1].mark_unhealthy()

    load_balancer = RoundRobinLoadBalancer(workers)

    # The unhealthy worker should be skipped.
    assert load_balancer.get_next_worker().worker_id == "worker-1"
    assert load_balancer.get_next_worker().worker_id == "worker-3"

    # Recover worker-2.
    workers[1].mark_healthy()

    # Worker-2 should be available again.
    assert load_balancer.get_next_worker().worker_id == "worker-1"
    assert load_balancer.get_next_worker().worker_id == "worker-2"
