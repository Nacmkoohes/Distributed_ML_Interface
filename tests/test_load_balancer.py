from load_balancer.round_robin import RoundRobinLoadBalancer


def test_round_robin_cycles_through_workers():
    workers = ["worker-1", "worker-2", "worker-3"]

    load_balancer = RoundRobinLoadBalancer(workers)

    assert load_balancer.get_next_worker() == "worker-1"
    assert load_balancer.get_next_worker() == "worker-2"
    assert load_balancer.get_next_worker() == "worker-3"
    assert load_balancer.get_next_worker() == "worker-1"