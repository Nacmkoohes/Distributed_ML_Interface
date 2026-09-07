class RoundRobinLoadBalancer:
    """
    Distributes requests across worker services
    using round-robin scheduling.
    """

    def __init__(self, workers):
        self.workers = workers
        self.current_index = 0

    def get_next_worker(self):
        worker = self.workers[self.current_index]

        self.current_index = (
            self.current_index + 1
        ) % len(self.workers)

        return worker