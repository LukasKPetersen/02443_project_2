import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from multi_ipp import MultiIPP

np.random.seed(42)

# We set the analysis base configuration
base_config = {
    'num_sources': 5,
    'lambda_on': 3,
    'omega_on': 1.5,
    'omega_off': 0.5,
    'num_servers': 2,
    'mu': 2,
    'queue_capacity': 3,
    'simulation_time': 1000
}

# We define the ranges for the sensitivity analysis
param_ranges = {
    'lambda_on':        np.linspace(0.01, 10, 10),
    'omega_on':         np.linspace(0.01, 10, 10),
    'omega_off':        np.linspace(0.01, 6, 10),
    'num_servers':      range(1, 8),
    'mu':               np.linspace(0.01, 5, 10),
    'queue_capacity':   [1, 2, 3, 5, 10, 20, np.inf]
}

# Run sensitivity analysis for each parameter and save the results
results = defaultdict(list)
for param_name, values in param_ranges.items():
    print(f"\nRunning sensitivity analysis for {param_name}...")
    
    # We iterate over the values to be compared
    for value in tqdm(values):
        config = base_config.copy()
        config[param_name] = value
        
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
        
        # Read metrics from the simulation log
        df = pd.DataFrame(mipp.event_log)
        arrival_events = df[df['event'] == 'arrival']
        
        if not arrival_events.empty:
            blocking_prob = arrival_events['blocked'].mean() if 'blocked' in arrival_events else 0
            avg_queue = arrival_events['queue_length'].mean() if 'queue_length' in arrival_events else 0
            avg_wait = arrival_events['waiting_time'].mean() if 'waiting_time' in arrival_events else 0
            
            results[param_name].append({
                'value': value,
                'blocking_prob': blocking_prob,
                'avg_queue': avg_queue,
                'avg_wait': avg_wait,
            })

# We plot the results
for param_name, data in results.items():
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=36)

    ax.plot(df['value'], df['blocking_prob'], 'o-', label='Blocking Probability')
    ax.plot(df['value'], df['avg_queue'], 's-', label='Average Queue Length')
    ax.plot(df['value'], df['avg_wait'], '^-', label='Average Waiting Time')

    ax.set_xlabel(param_name, fontsize=20)
    ax.set_ylabel('Metric Value', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True)
    ax.legend(fontsize=18)

    plt.tight_layout()
    plt.savefig(f'./graphics/sensitivity_{param_name}.png')
    plt.close(fig)
