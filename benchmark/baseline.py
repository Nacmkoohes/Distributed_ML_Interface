import  time


import  requests
import statistics
from concurrent.futures import ThreadPoolExecutor

URL='http://localhost:8000/predict'

payload={
    'user_id':1,
    'movie_id':10,
}

NUM_REQUESTS=1000
MAX_WORKERS=50

def send_request():
    start=time.perf_counter()
    response = requests.post(
        URL,
        json=payload,
    )
    end=time.perf_counter()
    latency = (end - start) * 1000
    return latency

start_time=time.perf_counter()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures=[
        executor.submit(send_request)
        for _ in range(NUM_REQUESTS)
    ]
latencies=[
    future.result()
    for future in futures
]
end_time=time.perf_counter()
total_time= end_time - start_time


average=sum(latencies)/len(latencies)
p50=statistics.median(latencies)
p95=statistics.quantiles(latencies,n=100)[94]
p99=statistics.quantiles(latencies,n=100)[98]
throughput=NUM_REQUESTS/total_time


print(f"\nAverage latency: {average:.2f} ms")
print(f"P50 latency: {p50:.2f} ms")
print(f"P95 latency: {p95:.2f} ms")
print(f"P99 latency: {p99:.2f} ms")
print(f"Throughput: {throughput:.2f} requests/second")