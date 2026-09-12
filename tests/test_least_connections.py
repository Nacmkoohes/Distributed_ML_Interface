from unittest.mock import patch

from load_balancer.least_connections import LeastConnectionsLoadBalancer


WORKERS = [
    "http://worker1:8000",
    "http://worker2:8000",
    "http://worker3:8000",
]


def test_selects_worker_with_fewest_connections():
    load_balancer = LeastConnectionsLoadBalancer(WORKERS)

    with patch.object(
        load_balancer,
        "is_healthy",
        return_value=True,
    ):
        assert load_balancer.get_next_worker() == WORKERS[0]


def test_selects_less_busy_worker():
    load_balancer = LeastConnectionsLoadBalancer(WORKERS)

    load_balancer.start_request(WORKERS[0])

    with patch.object(
        load_balancer,
        "is_healthy",
        return_value=True,
    ):
        assert load_balancer.get_next_worker() == WORKERS[1]


def test_selects_worker_with_fewest_connections():
    load_balancer = LeastConnectionsLoadBalancer(WORKERS)

    load_balancer.connections[WORKERS[0]] = 3
    load_balancer.connections[WORKERS[1]] = 1
    load_balancer.connections[WORKERS[2]] = 2

    with patch.object(
        load_balancer,
        "is_healthy",
        return_value=True,
    ):
        assert load_balancer.get_next_worker() == WORKERS[1]


def test_finish_request_decreases_connections():
    load_balancer = LeastConnectionsLoadBalancer(WORKERS)

    load_balancer.start_request(WORKERS[0])
    assert load_balancer.connections[WORKERS[0]] == 1

    load_balancer.finish_request(WORKERS[0])
    assert load_balancer.connections[WORKERS[0]] == 0


def test_worker_becomes_available_after_request_finishes():
    load_balancer = LeastConnectionsLoadBalancer(WORKERS)

    load_balancer.start_request(WORKERS[0])

    with patch.object(
        load_balancer,
        "is_healthy",
        return_value=True,
    ):
        assert load_balancer.get_next_worker() == WORKERS[1]

    load_balancer.finish_request(WORKERS[0])

    with patch.object(
        load_balancer,
        "is_healthy",
        return_value=True,
    ):
        assert load_balancer.get_next_worker() == WORKERS[0]

def test_unhealthy_worker_is_ignored():
    load_balancer = LeastConnectionsLoadBalancer(WORKERS)

    # Worker 1 is the least busy, but unhealthy.
    load_balancer.connections[WORKERS[0]] = 0
    load_balancer.connections[WORKERS[1]] = 2
    load_balancer.connections[WORKERS[2]] = 3

    def health_status(worker):
        return worker != WORKERS[0]

    with patch.object(
        load_balancer,
        "is_healthy",
        side_effect=health_status,
    ):
        assert load_balancer.get_next_worker() == WORKERS[1]

def test_raises_error_when_no_workers_are_healthy():
    load_balancer = LeastConnectionsLoadBalancer(WORKERS)

    with patch.object(
        load_balancer,
        "is_healthy",
        return_value=False,
    ):
        try:
            load_balancer.get_next_worker()
            assert False
        except RuntimeError as error:
            assert str(error) == "No healthy workers available"
def test_worker_is_used_again_after_recovery():
    load_balancer = LeastConnectionsLoadBalancer(WORKERS)

    load_balancer.connections[WORKERS[0]] = 0
    load_balancer.connections[WORKERS[1]] = 1
    load_balancer.connections[WORKERS[2]] = 2

    health = {
        WORKERS[0]: False,
        WORKERS[1]: True,
        WORKERS[2]: True,
    }

    def health_status(worker):
        return health[worker]

    with patch.object(
        load_balancer,
        "is_healthy",
        side_effect=health_status,
    ):
        # Worker 1 is unhealthy, so worker 2 should be selected.
        assert load_balancer.get_next_worker() == WORKERS[1]

        # Worker 1 recovers.
        health[WORKERS[0]] = True

        # Worker 1 has the fewest connections, so it should be selected.
        assert load_balancer.get_next_worker() == WORKERS[0]