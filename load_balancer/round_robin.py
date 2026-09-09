from idlelib.debugobj_r import remote_object_tree_item

import requests


class RoundRobinLoadBalancer:
    """
    Distributes requests across worker services
    using round-robin scheduling.
    """

    def __init__(self, workers):
        self.workers = workers
        self.current_index = 0

    def is_healthy(self, worker_url):
        try:
            response=requests.get(
                f"{worker_url}/health",
                timeout=1
            )
            return response.status_code==200
        except requests.RequestException:
            return False


    def get_next_worker(self):
        for _ in range(len(self.workers)):

            worker = self.workers[self.current_index]

            self.current_index = (
                self.current_index + 1
            ) % len(self.workers)

            if self.is_healthy(worker):
                return  worker

        raise RuntimeError('No healthy workers available!')
