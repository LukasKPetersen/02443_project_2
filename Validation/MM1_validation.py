import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import iv
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.insert(1, '../src')
import multi_ipp as MPP

# --- Theoretical M/M/1 PMF function ---
def mm1_pmf(k, t, i, lam, mu, summation_terms=100):
    k = np.atleast_1d(k)
    rho = lam / mu
    a = 2 * np.sqrt(lam * mu)
    exp_term = np.exp(-(lam + mu) * t)

    p_vals = []
    for k_val in k:
        term1 = rho**((k_val - i)/2) * iv(k_val - i, a * t)
        term2 = rho**((k_val - i - 1)/2) * iv(k_val + i + 1, a * t)

        j_start = k_val + i + 2
        js = np.arange(j_start, j_start + summation_terms)
        summation = np.sum(rho**(-js / 2) * iv(js, a * t))

        pk = exp_term * (term1 + term2 + (1 - rho) * rho**k_val * summation)
        p_vals.append(pk)

    return np.array(p_vals) if len(p_vals) > 1 else p_vals[0]

# --- Run simulation multiple times and bin queue lengths ---
def run_simulation(n_runs, sim_time, bin_width, lam, mu, omega_on=1, omega_off=1e8):
    bins = np.arange(0, sim_time + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    queue_hist = defaultdict(list)

    for _ in tqdm(range(n_runs), desc="Running simulations"):
        sim = MPP.MultiIPP(1, lam, omega_on, omega_off, 1, mu)
        sim.simulate_until(sim_time)
        df = pd.DataFrame(sim.event_log)

        # Drop NA values
        mask = ~df['num_in_system'].isna()
        times = df['time'].values[mask]
        num_in_system = df['num_in_system'].values[mask].astype(int)  # Convert nullable Int to int

        # Bin times and accumulate queue lengths
        bin_indices = np.digitize(times, bins) - 1  # 0-indexed
        for idx, k in zip(bin_indices, num_in_system):
            if 0 <= idx < len(bin_centers):
                queue_hist[idx].append(k)

    return queue_hist, bin_centers
# --- Build empirical distributions from histograms ---
def build_empirical_dists(queue_hist):
    empirical_dists = {}
    max_k = max((max(v) if v else 0) for v in queue_hist.values())
    for idx, values in queue_hist.items():
        counts = np.bincount(values, minlength=max_k + 1)
        probs = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
        empirical_dists[idx] = probs
    return empirical_dists

# --- Plot empirical vs. theoretical PMF ---
def plot_comparison(bin_centers, empirical_dists, lam, mu, i=0, selected_bins=[5, 15, 30]):
    for idx in selected_bins:
        if idx not in empirical_dists:
            print("here")
            continue
        empirical = empirical_dists[idx]
        k_vals = np.arange(len(empirical))
        t = bin_centers[idx]
        theoretical = mm1_pmf(k_vals, t, i, lam, mu)

        plt.figure(figsize=(8, 4))
        bar_width = 0.8

        # Plot empirical histogram as bars
        plt.bar(k_vals-1, empirical, width=bar_width, color='skyblue', alpha=0.7, label='Empirical')

        # Plot theoretical as dashed line with markers
        plt.plot(k_vals, theoretical, 'C1--', marker='x', label='Theoretical (M/M/1)', linewidth=1.5)

        plt.title(f"Queue Length PMF at Time ≈ {t:.1f}")
        plt.xlabel("Queue length k")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"test{idx}")
        plt.close()


# --- Parameters ---
n_runs = 1000
sim_time = 100
bin_width = 1.0
omega_on = 1  # IPP lambda
omega_off = 1e8    # IPP mu

lam_mm1 = 1
mu_mm1 = 0.5

# --- Run Everything ---
queue_hist, bin_centers = run_simulation(n_runs, sim_time, bin_width,lam_mm1,mu_mm1)
empirical_dists = build_empirical_dists(queue_hist)
plot_comparison(bin_centers, empirical_dists, lam=lam_mm1, mu=mu_mm1, i=0, selected_bins=[15,30,45,60,75,90])
