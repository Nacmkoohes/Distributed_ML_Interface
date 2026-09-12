import requests

class LeastConnectionsLoadBalancer:


    def __init__(self,workers):
        self.workers=workers

        self.connections={
            worker:0
            for worker in workers
        }
    def is_healthy(self,worker_url):
        try:
            response=requests.get(
                f"{worker_url}/health",
                timeout=1,
            )
            return response.status_code==200
        except requests.RequestException:
            return False

    def get_next_worker(self):
        healthy_workers=[
            worker
            for worker in self.workers
            if self.is_healthy(worker)
        ]


        if not healthy_workers:
            raise RuntimeError("No healthy workers available")


        return min(
            healthy_workers,
            #self.connections.get goes to values and find the minimum
            key=self.connections.get,
        )

    def start_request(self,worker):
        self.connections[worker]+=1

    def finish_request(self,worker):
        self.connections[worker]-=1


