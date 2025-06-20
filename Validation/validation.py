import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import iv
from tqdm import tqdm
import sys

sys.path.insert(1, '../src')
import multi_ipp as MPP
from scipy.stats import chisquare

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


def run_peek_histogram(n_runs, lam, mu, T, sim_time=100, omega_on=0.1, omega_off=1e8):
    peek_nums = []
    for _ in tqdm(range(n_runs), desc="Simulating"):
        sim = MPP.MultiIPP(1, lam, omega_on, omega_off, 1, mu)
        sim.simulate_until(sim_time, [T])
        df = pd.DataFrame(sim.event_log)
        peek_df = df[df['event'] == 'peek']
        if not peek_df.empty:
            peek_num = int(peek_df['num_in_system'].values[0])
            peek_nums.append(peek_num)

    peek_nums = np.array(peek_nums)
    max_val = int(peek_nums.max()) + 1
    hist_vals = np.histogram(peek_nums, bins=np.arange(max_val + 1))[0]
    k_vals = np.arange(max_val)
    return k_vals, hist_vals


def plot_empirical_vs_theoretical(k_vals, hist_vals, T, lam, mu, i=0):
    empirical = hist_vals / np.sum(hist_vals)
    theoretical = mm1_pmf(k_vals, T, i, lam, mu)

    plt.figure(figsize=(8, 4))
    plt.bar(k_vals, empirical, color='skyblue', alpha=0.7, label="Empirical")
    plt.plot(k_vals, theoretical, '-o', color="orange", label="Theoretical (M/M/1)")
    plt.title(f"Queue Length PMF at Time t = {T}")
    plt.xlabel("Queue length k")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pmf_plot_time{T}.pdf")
    plt.close()
    #plt.show()

def compute_ks_p_value(k_vals, hist_vals, T, lam, mu, i=0):
    # Normalize empirical histogram to probabilities
    N = np.sum(hist_vals)
    
    theo_props = mm1_pmf(k_vals, T, i, lam, mu)
    theo_props /= np.sum(theo_props)
    theoretical_sample=theo_props*N

    
    stat, p_value = chisquare(hist_vals, theoretical_sample)
    print(p_value)
    return p_value
def ks_test_plot(T,lam,mu,i=0,num_iter=100):
    p_vals=[]
    for iter in range(num_iter):
        print(iter)
        k_vals,hist_vals=run_peek_histogram(100,lam,mu,T)
        p_val=compute_ks_p_value(k_vals,hist_vals,T,lam,mu)
        p_vals.append(p_val)
    plt.hist(p_vals,bins=10,range=(0,1))
    plt.title(f"p-values for chi_square test at time {T}. Number of runs: {num_iter}")
    plt.savefig(f"chi_square_pvals_time_{T}")
    plt.show()
# Parameters
if False:
    for T in np.arange(20):
        lam = 1
        mu = 2
        #T = 1
        n_runs = 100
        # Run and plot
        k_vals, hist_vals = run_peek_histogram(n_runs, lam, mu, T)
        plot_empirical_vs_theoretical(k_vals, hist_vals, T, lam, mu)
T=15
lam=1
mu=2
k_vals, hist_vals = run_peek_histogram(1000, lam, mu, T)
plot_empirical_vs_theoretical(k_vals, hist_vals, T, lam, mu)
#ks_test_plot(T,lam,mu,num_iter=100)