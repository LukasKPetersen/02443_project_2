import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from tqdm import tqdm
import pandas as pd
import heapq


class IPPSource:
    """Single IPP source for MultiIPP."""
    def __init__(self, lambda_on, omega_on, omega_off, source_id):
        self.lambda_on = lambda_on
        self.omega_on = omega_on
        self.omega_off = omega_off
        self.state = False # False -> 'OFF', True -> 'ON'
        self.time = 0
        self.source_id = source_id
        self.arrival_dist = sps.expon(scale=1/lambda_on)
        self.sojourn_dists = {
            True: sps.expon(scale=1/omega_on),
            False: sps.expon(scale=1/omega_off)
        }
        self.event_log = []

    def step(self):
        """Advance to next ON/OFF period, return future arrivals."""
        events = []
        duration = self.sojourn_dists[self.state].rvs()
        t_start = self.time
        t_end = self.time + duration

        if self.state:
            t = t_start
            while True:
                t += self.arrival_dist.rvs()
                if t >= t_end:
                    break
                events.append({
                    'time': t,
                    'event': 'arrival',
                    'source_id': self.source_id
                })

        self.time = t_end
        self.state = not self.state
        events.append({
            'time': self.time,
            'event': 'state_change',
            'state': self.state,
            'source_id': self.source_id
        })
        return events

class IPPServers:
    """Class representing the servers for MultiIPP."""
    def __init__(self, num_servers, mu, queue_capacity=np.inf):
        self.num_servers = num_servers
        self.mu = mu
        self.server_busy_until = np.zeros(num_servers)
        self.departure_heap = []
        self.service_dist = sps.expon(scale=1/mu)
        self.queue_capacity = queue_capacity

    def handle_arrival(self, arrival):
        t = arrival['time']
        self.update_heap(t)

        queue_length = max(0, len(self.departure_heap) - np.sum(self.server_busy_until > t))
        
        if np.all(self.server_busy_until > t) and queue_length >= self.queue_capacity:
            event = {
                **arrival,
                'queue_length': queue_length,
                'blocked': True
            }
            return event
        
        
        i_soonest = np.argmin(self.server_busy_until)
        available_time = self.server_busy_until[i_soonest]
        start_time = max(t, available_time)
        service_time = self.service_dist.rvs()
        departure_time = start_time + service_time
        self.server_busy_until[i_soonest] = departure_time

        heapq.heappush(self.departure_heap, departure_time)
        event = {
            **arrival,
            'server_id': i_soonest,
            'start_service': start_time,
            'departure': departure_time,
            'waiting_time': start_time - t,
            'queue_length': max(0, len(self.departure_heap) - np.sum(self.server_busy_until > t)),
            'blocked': False,
            'num_in_system': len(self.departure_heap)
        }
        return event

    def update_heap(self, time):
        while self.departure_heap and self.departure_heap[0] <= time:
            heapq.heappop(self.departure_heap)
    
    def get_num_in_system(self, time):
        self.update_heap(time)
        return len(self.departure_heap)
    def get_q_length(self,time):
        self.update_heap(time)
        return max(0, len(self.departure_heap) - np.sum(self.server_busy_until > time))

class MultiIPP:
    """Multiple IPPs in parallel with multiple servers handling the arrivals. Supports infinite, finite, and no waiting room for queue."""
    def __init__(self, num_sources, lambda_on, omega_on, omega_off, num_servers, mu, queue_capacity=np.inf):
        """
        num_sources: number of superpositioned IPPs
        lambda_on: Poisson rate during 'ON' state
        omega_on: transition rate from 'ON' to 'OFF'
        omega_off: transition rate from 'OFF' to 'ON'
        mu: service rate
        """
        # for the sources:allow for passing different rate parameters as array while
        # still being able to pass a scalar to have them constant
        def scalar_to_full_array(scalar, size):
            if not isinstance(scalar, np.ndarray) or scalar.size == 1:
                return np.full(size, scalar)
        lambda_on = scalar_to_full_array(lambda_on, num_sources)
        omega_on = scalar_to_full_array(omega_on, num_sources)
        omega_off = scalar_to_full_array(omega_off, num_sources)
        
        assert lambda_on.size == num_sources
        assert omega_on.size == num_sources
        assert omega_off.size == num_sources
        
        self.sources = [IPPSource(lambda_on[i], omega_on[i], omega_off[i], i) for i in range(num_sources)]
        self.servers = IPPServers(num_servers, mu, queue_capacity)
        self.global_time = 0
        self.event_log = []

    def simulate_until(self, time_end, peek_times=None):
        """
        time_end: how long to simulate
        peek_times: array of times to peek at the system. Currenly observes 'num_in_system'.
        """
        if peek_times is None:
            future_events = []
        else:
            future_events = [{'time': t, 'event': 'peek'} for t in peek_times]
        
        for src in self.sources:
            future_events += src.step()

        while future_events:
            future_events.sort(key=lambda e: e['time'])
            event = future_events.pop(0)
            self.global_time = event['time']

            if self.global_time > time_end:
                break

            if event['event'] == 'arrival':
                event = self.servers.handle_arrival(event)
            elif event['event'] == 'state_change':
                src = self.sources[event['source_id']]
                future_events += src.step()
            elif event['event'] == 'peek':
                event['num_in_system'] = self.servers.get_num_in_system(event['time'])
                event['queue_length'] = self.servers.get_q_length(event['time'])
            self.event_log.append(event)


def test_MultiIPP():
    print('=== test_MultiIPP ===')
    np.random.seed(42)
    
    mipp = MultiIPP(num_sources=5, lambda_on=3, omega_on=1.5, omega_off=0.5, num_servers=2, mu=2, queue_capacity=3)
    mipp.simulate_until(100)
    
    df = pd.DataFrame(mipp.event_log)
    cols = ['source_id', 'server_id', 'queue_length']
    df = df.astype({col: 'Int64' for col in cols if col in df})
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].round(3)
    
    # print(df.to_string())

#if __name__ == '__main__':
#    test_MultiIPP()


def give_res(NSources,l_on, omega1,omega2,Nservers,mu,q_cap,simtime,burn_in,peek_times=None):
    np.random.seed(42)
    mipp = MultiIPP(num_sources=NSources, lambda_on=l_on, omega_on=omega1, omega_off=omega2, num_servers=Nservers, mu=mu, queue_capacity=q_cap)
    mipp.simulate_until(simtime,peek_times)
    print(f"Simulating system for {simtime}")
    mipp.event_log = [e for e in mipp.event_log if e['time'] > burn_in and e.get('event') == 'arrival']
    
    waiting_times = np.array([e.get('waiting_time', 0) for e in mipp.event_log if e.get('event') == 'arrival'])
    
    queue_times = np.array([e['time'] for e in mipp.event_log if 'queue_length' in e])
    queue_lengths = np.array([e['queue_length'] for e in mipp.event_log if 'queue_length' in e])
    blocked_count = np.array(sum(1 for e in mipp.event_log if e.get('blocked')))
    total_arrivals = np.array(sum(1 for e in mipp.event_log if e.get('event') == 'arrival'))
    #peeks = np.array([e['queue_length'] for e in mipp.event_log if 'queue_length' in e])
    
    wait_prop = sum(waiting_times > 0)/total_arrivals
    
    df = pd.DataFrame(mipp.event_log)
    cols = ['source_id', 'server_id', 'queue_length']
    df = df.astype({col: 'Int64' for col in cols if col in df})
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].round(3)

    df["blocked"].sum()
    with open('multi_ipp.txt', 'w') as f:
        f.write(df.to_string())
    
    # plots
    plt.figure()
    plt.hist(waiting_times[waiting_times > 0], bins=100,color="green",edgecolor="black",alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title('Distribution of waiting times')
    plt.grid(True)
    plt.savefig("Waitdist.pdf")
    plt.show()


    plt.figure()
    plt.step(queue_times, queue_lengths, where='post')
    plt.xlabel('Time')
    plt.ylabel('Queue length')
    plt.title('Queue length over time')
    plt.grid(True)
    plt.savefig("Qlength.pdf")
    plt.show()

    avg_wait = np.mean(waiting_times) if len(waiting_times) > 0 else 0
    block_prop = blocked_count / total_arrivals if total_arrivals > 0 else 0
    avg_qlength = np.mean(queue_lengths)
    
    print(f"Average wait time: {avg_wait:.3f}")
    print(f"Blocking proportion: {block_prop:.3f}")
    print(f"Fraction of people that had to wait: {wait_prop:.3f}")
    print(f"Average queue length per arrival: {avg_qlength:.3f}")
    return avg_wait, block_prop, wait_prop, avg_qlength

#Test parameters:
peek_times = np.arange(4200)
NSources = 4
Nservers = 1
l_on = 2
omega1 = 0.8
omega2 = 0.2
mu = 1.7777777777777783
q_cap = 30
simtime = 4_200
give_res(NSources,l_on, omega1,omega2,Nservers,mu,q_cap,simtime,200,peek_times)
N = 5
Nservers = np.arange(1,N+1)
avg_wait_N = np.empty(N)
block_prop_N = np.empty(N)
wait_prop_N = np.empty(N)




NUM_SOURCES = 4
NUM_SERVERS = 1
LAMBDA_ON = 2  # arrival rate when the source is ON
OMEGA_ON = 0.8  # state change rate from ON to OFF
OMEGA_OFF = 0.2  # state change rate from OFF to ON
ARRIVAL_TO_SERVICE_RATE_RATIO = 10/9
QUEUE_CAPACITY = 30  # maximum number of jobs in the queue
HORIZON = 4000
BURN_IN = 100  # initial time to ignore for steady state


def get_hyp_exp_params(omega_on, omega_off, lambda_on):
    t = (lambda_on + omega_on + omega_off)
    d = (t**2 - 4*lambda_on*omega_off)**0.5
    p1 = 1/2*(lambda_on-omega_on-omega_off + d)/d
    p2 = 1-p1
    gamma_1 = 1/2*(t + d)
    gamma_2 = 1/2*(t - d)
    mean_inter_arrival_time = p1/gamma_1 + p2/gamma_2

    return p1, p2, gamma_1, gamma_2, mean_inter_arrival_time

def get_service_rate(
        omega_on, 
        omega_off, 
        lambda_on, 
        arrival_to_service_rate_ratio, 
        num_sources, 
        num_servers
        ):

    mean_inter_arrival_rate = 1/get_hyp_exp_params(omega_on, omega_off, lambda_on)[-1]
    service_rate = mean_inter_arrival_rate * arrival_to_service_rate_ratio * num_sources / num_servers
    return service_rate

NUM_SOURCES = 4
NUM_SERVERS = 1
LAMBDA_ON = 2  # arrival rate when the source is ON
OMEGA_ON = 0.8  # state change rate from ON to OFF
OMEGA_OFF = 0.2  # state change rate from OFF to ON
ARRIVAL_TO_SERVICE_RATE_RATIO = 10/9
QUEUE_CAPACITY = 30  # maximum number of jobs in the queue
HORIZON = 4000
BURN_IN = 100  # initial time to ignore for steady state
get_service_rate(OMEGA_ON,OMEGA_OFF,LAMBDA_ON,ARRIVAL_TO_SERVICE_RATE_RATIO,NUM_SOURCES,NUM_SERVERS)

np.random.seed(42)
N = 100
horizon = 4200
burn_in = 200
peek_times = np.arange(burn_in + 1, horizon + 1)

params = {
    'num_sources': 4,
    'lambda_on': 2,
    'omega_on': 0.8,
    'omega_off': 0.2,
    'num_servers': 1,
    'mu': 1.7777777777777783,
    'queue_capacity': 30
}
vals = []
for i in range(N):
    mipp = MultiIPP(**params)
    mipp.simulate_until(horizon, peek_times)

    qlens = np.array([e['queue_length']
                    for e in mipp.event_log
                    if e.get('event') == 'arrival'])
    vals.append(qlens)
    print(i)
    
qlens = np.concatenate(vals)

plt.figure()
bins = np.arange(qlens.min(), qlens.max() + 2) - 0.5
plt.hist(qlens, bins=bins, density=True,color="green",edgecolor="black")
plt.xlabel('Queue length k')
plt.ylabel('Probability')
plt.title('Queue length distribution upon arrival')
plt.xticks(np.arange(0, qlens.max() + 1, 2))
plt.grid(True)
plt.savefig("Qdist_2.pdf")
plt.show()




from scipy.optimize import minimize
from scipy.stats import kstest
import numpy as np


arrival_times = [e['time']
                 for e in mipp.event_log
                 if e['event'] == 'arrival']


interarrivals = np.diff(arrival_times)

def hyperexp_pdf(x, p, l1, l2):
    return p * l1 * np.exp(-l1 * x) + (1 - p) * l2 * np.exp(-l2 * x)
omega_on = 0.4
omega_off = 0.3
lambda_on = 1
p = 1/2*((lambda_on-omega_on-omega_off)+np.sqrt((lambda_on+omega_on+omega_off)**2-4*lambda_on*omega_off))/(np.sqrt((lambda_on+omega_on+omega_off)**2-4*lambda_on*omega_off))
l1 = 1/2*((lambda_on+omega_on+omega_off)+np.sqrt((lambda_on+omega_on+omega_off)**2-4*lambda_on*omega_off))
l2 =  1/2*((lambda_on+omega_on+omega_off)-np.sqrt((lambda_on+omega_on+omega_off)**2-4*lambda_on*omega_off))

plt.hist(interarrivals, bins=400, density=True, alpha=0.6, label="Empirical",color="green",edgecolor="black")

x = np.linspace(0, max(interarrivals), 1000)
plt.plot(x, hyperexp_pdf(x, p, l1, l2), 'r-', label="Hyperexponential pdf")

plt.xlim((0,20))
plt.xlabel("Interarrival time")
plt.ylabel("Density")
plt.title("Single iPP interarrival time distribution")
plt.legend()
plt.grid(True)  
plt.savefig("Hyperexp.pdf")
plt.show()


def hyperexp_cdf(x, p, l1, l2):
    return p * (1 - np.exp(-l1 * x)) + (1 - p) * (1 - np.exp(-l2 * x))

from scipy.stats import kstest
N = 1000
p_vals = np.zeros(N)
np.random.seed(42)
cdf_fn = lambda x: hyperexp_cdf(x, p, l1, l2)
for i in range(N):
    omega_on = 0.4
    omega_off = 0.3
    lambda_on = 1
    mipp = MultiIPP(num_sources=1, lambda_on=lambda_on, omega_on=omega_on, omega_off=omega_off, num_servers=100_000, mu=1, queue_capacity=float("inf"))
    mipp.simulate_until(10_000)

    df = pd.DataFrame(mipp.event_log)
    cols = ['source_id', 'server_id', 'queue_length']
    df = df.astype({col: 'Int64' for col in cols if col in df})
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].round(3)

    arrival_times = [e['time']
                 for e in mipp.event_log
                 if e['event'] == 'arrival']

    interarrivals = np.diff(arrival_times)
    p = 1/2*((lambda_on-omega_on-omega_off)+np.sqrt((lambda_on+omega_on+omega_off)**2-4*lambda_on*omega_off))/(np.sqrt((lambda_on+omega_on+omega_off)**2-4*lambda_on*omega_off))
    l1 = 1/2*((lambda_on+omega_on+omega_off)+np.sqrt((lambda_on+omega_on+omega_off)**2-4*lambda_on*omega_off))
    l2 =  1/2*((lambda_on+omega_on+omega_off)-np.sqrt((lambda_on+omega_on+omega_off)**2-4*lambda_on*omega_off))
    values, counts = np.unique(interarrivals, return_counts=True)
    sorted_indices = np.argsort(values)
    values = values[sorted_indices]
    counts = counts[sorted_indices]
    empirical_cdf = np.cumsum(counts) / np.sum(counts)
    x = np.linspace(0, np.max(values), 1000)
    theoretical_cdf = hyperexp_cdf(x,p,l1,l2)
    D,ps = kstest(interarrivals,cdf_fn)
    p_vals[i] = ps
    print(i)



fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 

axs[0].hist(interarrivals, bins=400, density=True, alpha=0.6,
            label="Empirical distribution", color="green", edgecolor="black")

x = np.linspace(0, max(interarrivals), 1000)
axs[0].plot(x, hyperexp_pdf(x, p, l1, l2), 'r-', label="Hyperexponential pdf")

axs[0].set_xlim((0, 20))
axs[0].set_xlabel("Interarrival time")
axs[0].set_ylabel("Density")
axs[0].set_title("Single IPP interarrival time distribution")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(values, empirical_cdf, label="Empirical CDF", color="green")
axs[1].plot(x, theoretical_cdf, label="Theoretical CDF", color="red")
axs[1].set_xlabel("Interarrival time")
axs[1].set_ylabel("CDF")
axs[1].set_title("Empirical vs Theoretical CDF")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("IPP_interarrival_PDF_CDF.pdf")
plt.show()

plt.hist(p_vals,bins=25,density=True,color="green",edgecolor="black",alpha=0.7)
plt.xlabel("p-value")
plt.ylabel("Density")
plt.title("Distribution of p-values")
plt.grid(True)  
plt.savefig("p_vals.pdf")
plt.show()

