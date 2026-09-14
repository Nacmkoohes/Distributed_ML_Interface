import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark/results.csv")

for metric in [
    "average_latency_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "throughput_req_per_sec",
]:
    plt.figure()

    for strategy in df["strategy"].unique():
        data = df[df["strategy"] == strategy]

        plt.plot(
            data["concurrency"],
            data[metric],
            marker="o",
            label=strategy,
        )

    plt.xlabel("Concurrency")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Concurrency")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"benchmark/plots/{metric}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()