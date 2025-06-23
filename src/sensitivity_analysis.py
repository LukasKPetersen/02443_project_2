import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from multi_ipp import MultiIPP
from params import get_hyp_exp_params

np.random.seed(42)

# We set the analysis base configuration
lambda_on = 2
omega_on = 0.8
omega_off = 0.2
_, _, _, _, mean_inter_arrival_time = get_hyp_exp_params(omega_on, omega_off, lambda_on)

beta = 1/mean_inter_arrival_time
n = 4  # Number of sources
m = 1  # Number of servers

base_config = {
    'num_sources': n,
    'lambda_on': lambda_on,
    'omega_on': omega_on,
    'omega_off': omega_off,
    'num_servers': m,
    'mu': (10/9*n)/(beta*m),
    'queue_capacity': 30,
    'simulation_time': 1000
}

# We define the ranges for the sensitivity analysis
param_ranges = {
    'lambda_on':        np.linspace(0.01, 10, 10),
    'omega_on':         np.linspace(0.01, 10, 10),
    'omega_off':        np.linspace(0.01, 6, 10),
    'num_servers':      range(1, 8),
    'mu':               np.linspace(0.01, 5, 10),
    'queue_capacity':   [0, 1, 2, 3, 5, 7, 10, 20, np.inf]
}

# Run sensitivity analysis for each parameter and save the results
results = defaultdict(list)
for param_name, values in param_ranges.items():
    print(f"\nRunning sensitivity analysis for {param_name}...")
    
    # We iterate over the values to be compared
    for param_val in tqdm(values):
        config = base_config.copy()
        config[param_name] = param_val
        
        # Set up the simulator
        mipp = MultiIPP(
            num_sources=config['num_sources'],
            lambda_on=config['lambda_on'],
            omega_on=config['omega_on'],
            omega_off=config['omega_off'],
            num_servers=config['num_servers'],
            mu=config['mu'],
            queue_capacity=config['queue_capacity']
        )
        # We run the simulation
        mipp.simulate_until(config['simulation_time'])
        
        # Read metrics from the simulation log (filter for arrivals only)
        df = pd.DataFrame(mipp.event_log)
        arrival_events = df[df['event'] == 'arrival']
        
        # Store the result
        results[param_name].append({
            'param_val': param_val,
            'blocking_prob': arrival_events['blocked'].mean() if 'blocked' in arrival_events else 0,
            'avg_wait': arrival_events['waiting_time'].mean()
        })

# We plot the results
for param_name, data in results.items():
    df = pd.DataFrame(data)
    fig, axL = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=36)

    # Left y-axis for avg_wait
    axL.plot(df['param_val'], df['avg_wait'], '^-', label='Average Waiting Time')
    axL.set_xlabel(param_name, fontsize=20)
    axL.set_ylabel('Average Waiting Time', fontsize=20)
    axL.tick_params(axis='both', labelsize=18)
    axL.grid(True)

    # Right y-axis for blocking_prob
    axR = axL.twinx()
    axR.plot(df['param_val'], df['blocking_prob'], 'o-', color='tab:red', label='Blocking Probability')
    axR.set_ylabel('Blocking Probability', fontsize=20, color='tab:red')
    axR.tick_params(axis='y', labelsize=18, colors='tab:red')

    # Legends
    lines_1, labels_1 = axL.get_legend_handles_labels()
    lines_2, labels_2 = axR.get_legend_handles_labels()
    axL.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=18)

    plt.tight_layout()
    plt.savefig(f'./graphics/sensitivity_{param_name}.png')
    plt.close(fig)
