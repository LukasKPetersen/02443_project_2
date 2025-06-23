import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os
print(os.getcwd())
sys.path.insert(0, '.')
from src.multi_ipp import MultiIPP

def compute_hyperexp_params(lam, omega1, omega2):
    # Common terms
    A = lam + omega1 + omega2
    B = np.sqrt(A**2 - 4 * lam * omega2)

    # Parameters
    p1 = 0.5 * ((lam - omega1 - omega2 + B) / B)
    p2 = 1 - p1

    gamma1 = 0.5 * (A + B)
    gamma2 = 0.5 * (A - B)

    # Mean of the hyperexponential distribution
    mean = p1 / gamma1 + p2 / gamma2

    return p1,p2,gamma1,gamma2,mean
    
def test_MultiIPP():
    print('=== test_MultiIPP ===')
    np.random.seed(42)
    # mipp = MultiIPP(num_sources=5, lambda_on=3, omega_on=1.5, omega_off=0.5, num_servers=2, mu=2, queue_capacity=3)
    
    num_sources = 4
    num_servers = 1
    lam = 2
    omega_on = 0.8
    omega_off = 0.2
    mean_inter = compute_hyperexp_params(lam=lam,omega1=omega_on,omega2=omega_off)[4]
    mu = 10/9 * num_sources/(mean_inter*num_servers)
    print(mu)
    mipp = MultiIPP(
        num_sources=num_sources, lambda_on=lam,
        omega_on=omega_on, omega_off=omega_off,
        num_servers=num_servers, mu=mu,
        queue_capacity=50)
    mipp.simulate_until(1000)

    df = pd.DataFrame(mipp.event_log)
    cols = ['source_id', 'server_id', 'queue_length']
    df = df.astype({col: 'Int64' for col in cols if col in df})
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].round(3)
    
    block_prob = np.mean(df['blocked'])
    avg_queue_len = np.mean(df['queue_length'])
    avg_wait = np.mean(df['waiting_time'])
    print(f'{block_prob=}')
    print(f'{avg_queue_len=}')
    print(f'{avg_wait=}')
    
    with open('logs/multi_ipp.txt', 'w') as f:
        f.write(df.to_string())

def exp_server():
    np.random.seed(42)
    num_sources = 4
    num_servers = 1
    lam = 2
    omega_on = 0.8
    omega_off = 0.2
    mean_inter = compute_hyperexp_params(lam=lam,omega1=omega_on,omega2=omega_off)[4]
    mu = 10/9 * num_sources/(mean_inter*num_servers)
    print(mu)
    mipp = MultiIPP(
        num_sources=num_sources, lambda_on=lam,
        omega_on=omega_on, omega_off=omega_off,
        num_servers=num_servers, mu=mu,
        queue_capacity=50)
    mipp.simulate_until(1000)
    
    df = pd.DataFrame(mipp.event_log)
    block_prob = np.mean(df['blocked'])
    avg_queue_len = np.mean(df['queue_length'])
    avg_wait = np.mean(df['waiting_time'])
    
def experiment_varying_servers(num_reps):
    np.random.seed(42)

    num_sources = 4
    lam = 2
    omega_on = 0.8
    omega_off = 0.2
    server_range = np.logspace(0, 7, 8, base=2).astype(int)  # [1, 2, 4, ..., 128]
    service_factors = [1.5, 1.1, 1.01, 1.001]

    num_sf = len(service_factors)
    num_srv = len(server_range)

    block_probs = np.zeros((num_sf, num_srv, num_reps))
    queue_lens = np.zeros((num_sf, num_srv, num_reps))
    wait_times = np.zeros((num_sf, num_srv, num_reps))

    for idx_factor, service_factor in tqdm(enumerate(service_factors)):
        for idx_server, num_servers in tqdm(enumerate(server_range), leave=False):
            mean_inter = compute_hyperexp_params(
                lam=lam, omega1=omega_on, omega2=omega_off
            )[4]
            mu = service_factor * num_sources / (mean_inter * num_servers)

            for rep in range(num_reps):
                mipp = MultiIPP(
                    num_sources=num_sources,
                    lambda_on=lam,
                    omega_on=omega_on,
                    omega_off=omega_off,
                    num_servers=num_servers,
                    mu=mu,
                    queue_capacity=30
                )
                burn_in = 200
                mipp.simulate_until(4000 + burn_in)
                df = pd.DataFrame(mipp.event_log)
                df = df[df['time'] > burn_in]

                block_probs[idx_factor, idx_server, rep] = np.mean(df['blocked'])
                queue_lens[idx_factor, idx_server, rep] = np.mean(df['queue_length'])
                wait_times[idx_factor, idx_server, rep] = np.mean(df['waiting_time'])

    def plot_metric(data, ylabel, title):
        colors = ['red','blue','green','magenta']
        plt.figure(figsize=(8, 5))
        for idx_factor, sf in enumerate(service_factors):
            # Plot individual rep points
            # for idx_server, num_servers in enumerate(server_range):
            #     plt.scatter(
            #         [num_servers] * num_reps,
            #         data[idx_factor, idx_server, :],
            #         alpha=0.2,
            #         color=colors[idx_factor],
            #         s=6
            #     )
            # Plot mean line
            means = data[idx_factor].mean(axis=1)
            plt.plot(server_range, means, marker='o', label=f'Service Factor = {sf}', linewidth=2, color=colors[idx_factor])

        plt.xlabel('Number of Servers')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.xscale('log', base=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'exp/{ylabel}.png')
        plt.show()

    plot_metric(block_probs, 'Blocking Probability', 'Blocking Probability vs Number of Servers')
    plot_metric(queue_lens, 'Average Queue Length', 'Average Queue Length vs Number of Servers')
    plot_metric(wait_times, 'Average Waiting Time', 'Average Waiting Time vs Number of Servers')

if __name__ == '__main__':
    # test_MultiIPP()
    experiment_varying_servers(num_reps=30)