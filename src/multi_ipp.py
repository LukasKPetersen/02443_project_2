import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from tqdm import tqdm
import pandas as pd
import heapq


class CachedRVWrapper:
    def __init__(self, rv, cache_size=1000):
        self.rv = rv
        self.cache_size = cache_size
        self.reset()

    def reset(self):
        """Reset the cache and index."""
        self.cache = self.rv.rvs(size=self.cache_size)
        self.index = 0

    def rvs(self):
        if self.index >= self.cache_size:
            self.reset()
            
        result = self.cache[self.index]
        self.index += 1
        return result

class IPPSource:
    """Single IPP source for MultiIPP."""
    def __init__(self, lambda_on, omega_on, omega_off, source_id):
        self.lambda_on = lambda_on
        self.omega_on = omega_on
        self.omega_off = omega_off
        self.state = False # False -> 'OFF', True -> 'ON'
        self.time = 0
        self.source_id = source_id
        self.arrival_dist = CachedRVWrapper(sps.expon(scale=1/lambda_on))
        self.sojourn_dists = {
            True: CachedRVWrapper(sps.expon(scale=1/omega_on)),
            False: CachedRVWrapper(sps.expon(scale=1/omega_off))
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
    with open('logs/multi_ipp.txt', 'w') as f:
        f.write(df.to_string())

if __name__ == '__main__':
    test_MultiIPP()