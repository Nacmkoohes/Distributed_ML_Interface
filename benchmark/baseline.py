import time
import sys
import os
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor
import csv


URL = "http://localhost:8000/predict"

payload = {
    "user_id": 1,
    "movie_id": 10,
}

NUM_REQUESTS = 1000


NUM_RUNS = 3

# The first command-line argument defines concurrency.
#
# Example:
# python benchmark/baseline.py 5 round_robin
#
# MAX_WORKERS = 5
MAX_WORKERS = int(sys.argv[1])




def send_request():

    start = time.perf_counter()

    try:
        response = requests.post(
            URL,
            json=payload,
            timeout=10,
        )

        # Stop measuring time
        end = time.perf_counter()

        # Convert seconds to milliseconds
        latency = (end - start) * 1000

        return latency, response.status_code

    except requests.RequestException:
        # Even if the request fails,
        # we still record its latency.

        end = time.perf_counter()

        latency = (end - start) * 1000

        # Status code 0 means the request failed
        return latency, 0


# Run one complete benchmark


def run_benchmark():


    # Start total benchmark timer
    start_time = time.perf_counter()

    # ThreadPoolExecutor controls concurrency.
    #
    # If MAX_WORKERS = 5,
    # at most 5 requests are processed concurrently.
    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(send_request)
            for _ in range(NUM_REQUESTS)
        ]

    # Get the result of every request
    results = [
        future.result()
        for future in futures
    ]

    # Stop total benchmark timer
    end_time = time.perf_counter()

    total_time = end_time - start_time

    return results, total_time


# We keep the results of every run separately.

all_status_codes = []
all_results = []
all_total_times = []


for run in range(NUM_RUNS):

    # Run 1000 requests
    results, total_time = run_benchmark()

    # Save total time of this run.
    #
    # We need this later to calculate
    # throughput for each run.
    all_total_times.append(total_time)

    # Extract latency from each request
    latencies = [
        result[0]
        for result in results
    ]

    # Save this run's latencies
    all_results.append(latencies)

    # Extract HTTP status codes
    status_codes = [
        result[1]
        for result in results
    ]

    # Save this run's status codes
    all_status_codes.append(status_codes)




# Calculate average latency for each run
run_averages = [
    sum(latencies) / len(latencies)
    for latencies in all_results
]

# Then calculate the average of the 3 runs
average = (
    sum(run_averages)
    / len(run_averages)
)


# P50 = median latency.
#
# 50% of requests are faster than this value
# and 50% are slower.

run_p50 = [
    statistics.median(latencies)
    for latencies in all_results
]

# Average P50 across the 3 runs
p50 = (
    sum(run_p50)
    / len(run_p50)
)




# P95 means:
# 95% of requests finished at or below this latency.
#
# The slowest 5% are above it.

run_p95 = [
    statistics.quantiles(
        latencies,
        n=100
    )[94]
    for latencies in all_results
]

# Average P95 across the 3 runs
p95 = (
    sum(run_p95)
    / len(run_p95)
)



# P99 means:
# 99% of requests finished at or below this latency.
#
# The slowest 1% are above it.

run_p99 = [
    statistics.quantiles(
        latencies,
        n=100
    )[98]
    for latencies in all_results
]

# Average P99 across the 3 runs
p99 = (
    sum(run_p99)
    / len(run_p99)
)



# Throughput tells us:
#
# How many requests can the system process
# per second?
#
# Formula:
#
# throughput = number of requests / total time

run_throughputs = [
    NUM_REQUESTS / total_time
    for total_time in all_total_times
]

# Average throughput across the 3 runs
throughput = (
    sum(run_throughputs)
    / len(run_throughputs)
)




run_error_rates = []


for status_codes in all_status_codes:

    # A successful HTTP request is:
    # 200 <= status < 300
    #
    # Everything else is considered failed.

    failed = sum(
        status < 200 or status >= 300
        for status in status_codes
    )

    # Calculate error rate for this run
    error_rate = (
        failed / len(status_codes)
    ) * 100

    run_error_rates.append(error_rate)


# Average error rate across the 3 runs
error_rate = (
    sum(run_error_rates)
    / len(run_error_rates)
)




# We ran:
#
# 1000 requests × 3 runs = 3000 requests

successful_requests = sum(
    200 <= status < 300
    for statuses in all_status_codes
    for status in statuses
)

failed_requests = (
    NUM_REQUESTS * NUM_RUNS
    - successful_requests
)



# The second command-line argument tells us
# which strategy we are testing.
#
# Example:
#
# python benchmark/baseline.py 5 round_robin
#
# strategy = "round_robin"

strategy = sys.argv[2]


# --------------------------------------------------
# 13. Save results to CSV
# --------------------------------------------------

results_file = "benchmark/results.csv"

# Check whether the CSV already exists
file_exists = os.path.exists(results_file)


# Open CSV in append mode.
#
# This means new benchmark results are added
# without deleting previous experiments.

with open(
    results_file,
    "a",
    newline="",
) as file:

    writer = csv.writer(file)

    # If this is the first time creating the file,
    # write the column names.

    if not file_exists:
        writer.writerow([
            "strategy",
            "concurrency",
            "average_latency_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "throughput_req_per_sec",
            "error_rate",
        ])

    # Save ONE row for this configuration.
    #
    # The values here are the averages
    # across the 3 benchmark runs.

    writer.writerow([
        strategy,
        MAX_WORKERS,
        average,
        p50,
        p95,
        p99,
        throughput,
        error_rate,
    ])



print(
    f"\nAverage latency: "
    f"{average:.2f} ms"
)

print(
    f"P50 latency: "
    f"{p50:.2f} ms"
)

print(
    f"P95 latency: "
    f"{p95:.2f} ms"
)

print(
    f"P99 latency: "
    f"{p99:.2f} ms"
)

print(
    f"Throughput: "
    f"{throughput:.2f} requests/second"
)

print(
    f"Successful requests: "
    f"{successful_requests}"
)

print(
    f"Failed requests: "
    f"{failed_requests}"
)

print(
    f"Error rate: "
    f"{error_rate:.2f}%"
)