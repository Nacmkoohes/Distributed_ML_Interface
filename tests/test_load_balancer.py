import pytest
from unittest.mock import patch
from load_balancer.round_robin import RoundRobinLoadBalancer
from workers.worker import MLWorker

WORKERS = [
    "http://worker1:8000",
    "http://worker2:8000",
    "http://worker3:8000",
]

def test_round_robin_cycles_through_workers():

    load_balancer = RoundRobinLoadBalancer(WORKERS)
    with patch.object(
        load_balancer,
        "is_healthy",
        return_value=True
    ):
        assert load_balancer.get_next_worker() == WORKERS[0]
        assert load_balancer.get_next_worker() == WORKERS[1]
        assert load_balancer.get_next_worker() == WORKERS[2]
        assert load_balancer.get_next_worker() == WORKERS[0]


def test_round_robin_skips_unhealthy_workers():

    load_balancer = RoundRobinLoadBalancer(WORKERS)
    def health_status(worker_url):
        return  worker_url !=WORKERS[1]

    with patch.object(
        load_balancer,
        "is_healthy",
        side_effect=health_status,
    ):
        assert load_balancer.get_next_worker() == WORKERS[0]
        assert load_balancer.get_next_worker() == WORKERS[2]
        assert load_balancer.get_next_worker() == WORKERS[0]

def test_raises_error_when_no_workers_are_healthy():


    load_balancer=RoundRobinLoadBalancer(WORKERS)

    with patch.object(
        load_balancer,
        "is_healthy",
        return_value=False,
    ):

        with pytest.raises(RuntimeError, match="No healthy workers available"):
            load_balancer.get_next_worker()

def test_unhealthy_worker_can_recover():

    load_balancer = RoundRobinLoadBalancer(WORKERS)

    healthy_workers = {
        WORKERS[0]: True,
        WORKERS[1]: False,
        WORKERS[2]: True,
    }

    def health_status(worker_url):
        return healthy_workers[worker_url]

    with patch.object(
        load_balancer,
        "is_healthy",
        side_effect=health_status,
    ):
        # The unhealthy worker should be skipped.
        assert load_balancer.get_next_worker()== WORKERS[0]
        assert load_balancer.get_next_worker()== WORKERS[2]

    # Recover worker-2.
        healthy_workers[WORKERS[1]]=True

    # Worker-2 should be available again.
        assert load_balancer.get_next_worker() == WORKERS[0]
        assert load_balancer.get_next_worker() == WORKERS[1]

