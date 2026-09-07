class RoundRobinLoadBalancer:
    """
    Distributes requests across workers in a circular order.

    Each request is assigned to the next worker in the list.
    When the last worker is reached, selection starts again
    from the first worker.
    """
    def __init__(self,workers):
        self.workers=workers
        self.current_index=0
    def get_next_worker(self):

        for _ in range(len(self.workers)):
            worker=self.workers[self.current_index]

            self.current_index=(self.current_index +1)%len(self.workers)
            if worker.health_check():
                return  worker

        raise  RuntimeError('No healthy workers available')