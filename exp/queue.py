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

def controlify(target_variable, control_variable, mu): 

    cov = np.cov(target_variable, control_variable)[0, 1]
    control_variance = np.var(control_variable)
    c = - (cov / control_variance)

    controlled_target = [tv + c * (cv - mu) for tv, cv in zip(target_variable, control_variable)]
    return np.array(controlled_target)


def experiment_varying_servers(num_reps):
    np.random.seed(42)

    num_sources = 4
    lam = 2
    omega_on = 0.8
    omega_off = 0.2
    server_range = np.logspace(0, 7, 8, base=2).astype(int)  # [1, 2, 4, ..., 128]
    # server_range = np.logspace(0, 7, 3, base=2).astype(int)  # [1, 2, 4, ..., 128]
    service_factors = [1.5, 1.1, 1.01, 1.001]
    # service_factors = [1.5, 1.1]

    num_sf = len(service_factors)
    num_srv = len(server_range)

    block_probs = np.zeros((num_sf, num_srv, num_reps))
    queue_lens = np.zeros((num_sf, num_srv, num_reps))
    wait_times = np.zeros((num_sf, num_srv, num_reps))
    average_inter_arrival_times = np.zeros((num_sf, num_srv, num_reps))
    control_mu = np.zeros((num_sf, num_srv))

    for idx_factor, service_factor in enumerate(tqdm(service_factors)):
        for idx_server, num_servers in enumerate(tqdm(server_range, leave=False)):
            mean_inter = compute_hyperexp_params(
                lam=lam, omega1=omega_on, omega2=omega_off
            )[4]
            mu = service_factor * num_sources / (mean_inter * num_servers)
            control_mu[idx_factor, idx_server] = mean_inter/num_sources
            for rep in tqdm(range(num_reps), leave=False):
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
                mipp.simulate_until(4000 + burn_in, verbose=False)
                df = pd.DataFrame(mipp.event_log)
                df = df[df['time'] > burn_in]

                block_probs[idx_factor, idx_server, rep] = np.mean(df['blocked'])
                queue_lens[idx_factor, idx_server, rep] = np.mean(df['queue_length'])
                wait_times[idx_factor, idx_server, rep] = np.mean(df['waiting_time'])
                arrival_times = df["time"].values[df["event"] == "arrival"]
                average_inter_arrival_times[idx_factor, idx_server, rep] = np.mean(np.diff(arrival_times))
                # breakpoint()
                # raise ValueError("Simulated data contains NaN values. Check the simulation parameters and logic.")

    def plot_metric(data, ylabel, title):
        colors = ['red','blue','green','magenta']
        plt.figure(figsize=(8, 5))
        controlled_vars = []
        original_vars = []
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
            # controlled = data[idx_factor].mean(axis=1)
            controlled = [
                controlify(data[idx_factor, idx_n_serv], average_inter_arrival_times[idx_factor, idx_n_serv], control_mu[idx_factor, idx_n_serv])
                for idx_n_serv, sf in enumerate(server_range)
            ]
            controlled = np.array(controlled)
            original = data[idx_factor]
            plt.plot(server_range, controlled.mean(axis=1), marker='o', label=f'Service Factor = {sf}', linewidth=2, color=colors[idx_factor])
            controlled_vars.append(controlled.var(axis=1))
            upper_bound = controlled.mean(axis=1) + controlled.std(axis=1)/ np.sqrt(num_reps)* 1.96
            lower_bound = controlled.mean(axis=1) - controlled.std(axis=1)/ np.sqrt(num_reps)* 1.96
            plt.fill_between(server_range, lower_bound, upper_bound, alpha=0.2, color=colors[idx_factor])
            original_vars.append(original.var(axis=1))

        controlled_vars = np.array(controlled_vars)
        original_vars = np.array(original_vars)

        # breakpoint()
        variance_reductions = 1 - (np.array(controlled_vars) / np.array(original_vars))
        variance_reductions[original_vars == 0] = 0  # Avoid division by zero
        plt.xlabel('Number of Servers')
        plt.ylabel(ylabel)
        plt.title(title + "\n" + f"Variance Reduction = {variance_reductions.mean()*100:.1f}%")
        plt.grid(True)
        plt.xscale('log', base=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'exp/{ylabel}.png')
        plt.show()

    plot_metric(block_probs, 'Blocking Probability', 'Blocking Probability vs Number of Servers')
    plot_metric(queue_lens, 'Average Queue Length', 'Average Queue Length vs Number of Servers')
    plot_metric(wait_times, 'Average Waiting Time', 'Average Waiting Time vs Number of Servers')

 
def experiment_varying_queue_caps(num_reps):
    np.random.seed(42)

    num_sources = 4
    lam = 2
    omega_on = 0.8
    omega_off = 0.2
    service_factor = 1.1

    server_range = np.logspace(0, 7, 8, base=2).astype(int)  # [1, 2, 4, ..., 128]
    queue_capacities = [0, 2, 5, 10, 30]

    # num_sf = len(service_factors)
    num_srv = len(server_range)
    num_qc = len(queue_capacities)

    block_probs = np.zeros((num_qc, num_srv, num_reps))
    queue_lens = np.zeros((num_qc, num_srv, num_reps))
    wait_times = np.zeros((num_qc, num_srv, num_reps))
    time_in_system = np.zeros((num_qc, num_srv, num_reps))
    average_inter_arrival_times = np.zeros((num_qc, num_srv, num_reps))
    control_mu = np.zeros((num_qc, num_srv))

    for idx_qc, queue_capacity in enumerate(tqdm(queue_capacities)):
        for idx_server, num_servers in enumerate(tqdm(server_range, leave=False)):
            mean_inter = compute_hyperexp_params(
                lam=lam, omega1=omega_on, omega2=omega_off
            )[4]
            mu = service_factor * num_sources / (mean_inter * num_servers)
            control_mu[idx_qc, idx_server] = mean_inter/num_sources

            for rep in tqdm(range(num_reps), leave=False):
                mipp = MultiIPP(
                    num_sources=num_sources,
                    lambda_on=lam,
                    omega_on=omega_on,
                    omega_off=omega_off,
                    num_servers=num_servers,
                    mu=mu,
                    queue_capacity=queue_capacity
                )
                burn_in = 200
                mipp.simulate_until(4000 + burn_in)
                df = pd.DataFrame(mipp.event_log)
                df = df[df['time'] > burn_in]

                block_probs[idx_qc, idx_server, rep] = np.mean(df['blocked'])
                queue_lens[idx_qc, idx_server, rep] = np.mean(df['queue_length'])
                wait_times[idx_qc, idx_server, rep] = np.mean(df['waiting_time'])
                time_in_system[idx_qc, idx_server, rep] = np.mean(df["departure"] - df["start_service"])
                arrival_times = df["time"].values[df["event"] == "arrival"]
                average_inter_arrival_times[idx_qc, idx_server, rep] = np.mean(np.diff(arrival_times))

    def plot_metric(data, ylabel, title):
        colors = ['red','blue','green','magenta', 'orange']
        controlled_vars = []
        original_vars = []
        plt.figure(figsize=(8, 5))
        for idx_qc, queue_capacity in enumerate(queue_capacities):
            # Plot individual rep points
            # for idx_server, num_servers in enumerate(server_range):
            #     plt.scatter(
            #         [num_servers] * num_reps,
            #         data[idx_factor, idx_server, :],
            #         alpha=0.2,
            #         color=colors[idx_factor],
            #         s=6
            #     )
            # # Plot mean line
            # means = data[idx_qc].mean(axis=1)
            # plt.plot(server_range, means, marker='o', label=f'Queue Capacity = {queue_capacity}', linewidth=2, color=colors[idx_qc])

            controlled = [
                controlify(data[idx_qc, idx_n_serv], average_inter_arrival_times[idx_qc, idx_n_serv], control_mu[idx_qc, idx_n_serv])
                for idx_n_serv, sf in enumerate(server_range)
            ]
            controlled = np.array(controlled)
            original = data[idx_qc]
            # breakpoint()
            plt.plot(server_range, controlled.mean(axis=1), marker='o', label=f'Queue Capacity = {queue_capacity}', linewidth=2, color=colors[idx_qc])
            controlled_vars.append(controlled.var(axis=1))
            upper_bound = controlled.mean(axis=1) + controlled.std(axis=1)/ np.sqrt(num_reps)* 1.96
            lower_bound = controlled.mean(axis=1) - controlled.std(axis=1)/ np.sqrt(num_reps)* 1.96
            plt.fill_between(server_range, lower_bound, upper_bound, alpha=0.2, color=colors[idx_qc])
            original_vars.append(original.var(axis=1))

        controlled_vars = np.array(controlled_vars)
        original_vars = np.array(original_vars)

        # breakpoint()
        variance_reductions = 1 - (np.array(controlled_vars) / np.array(original_vars))
        variance_reductions[original_vars == 0] = 0  # Avoid division by zero


        plt.xlabel('Number of Servers')
        plt.ylabel(ylabel)
        plt.title(title + "\n" + f"Variance Reduction = {variance_reductions.mean()*100:.1f}%")
        plt.grid(True)
        # breakpoint()
        plt.xscale('log', base=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'exp/{ylabel}_queue_cap.png')
        plt.show()

    plot_metric(block_probs, 'Blocking Probability', 'Blocking Probability vs Number of Servers')
    plot_metric(time_in_system, 'Average Time in System', 'Average Time in System vs Number of Servers')




if __name__ == '__main__':
    # test_MultiIPP()
    # experiment_varying_servers(num_reps=30)
    plt.close('all')
    # plt.figure()
    experiment_varying_queue_caps(num_reps=30)