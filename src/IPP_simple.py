import numpy as np

class SingleServerIPP:
    """
    Implementation of a Single Server ON/OFF Interrupted Poisson Process (IPP)
    Based on the image: 02443_project_2/material/IPP_model.jpg
    """

    def __init__(self, lambda_on, gamma1, gamma2, init_state='ON', seed=42):
        self.lambda_on = lambda_on  # intensity
        self.gamma1 = gamma1  # rate of state transition from ON to OFF
        self.gamma2 = gamma2  # rate of state transition from OFF to ON
        self.state = init_state
        self.rng = np.random.default_rng(seed)
        self.next_event_time = 0
        self.next_arrival_time = self.rng.exponential(1/lambda_on) if init_state == 'ON' else float('inf')
        self.set_sojourn_time()

    def set_sojourn_time(self):
        transition_rate = self.gamma1 if self.state == 'ON' else self.gamma2
        self.curr_sojourn_time = self.next_event_time + self.rng.exponential(1/transition_rate)

    def generate_arrivals(self):
        """
        Whenever the state is ON, generate all arrivals until the next state change.
        """
        arrivals = []
        while self.next_event_time < self.curr_sojourn_time:
            arrivals.append(self.next_event_time)
            self.next_event_time += self.rng.exponential(1/self.lambda_on)
        return arrivals

    def run(self, run_time=10000):
        """
        Run the IPP process for a specified duration.
        """
        arrivals = []
        while self.next_event_time < run_time:
            if self.state == 'ON':
                arrivals.extend(self.generate_arrivals())
                self.state = 'OFF'
                self.next_event_time = self.curr_sojourn_time
            else:
                self.state = 'ON'
                self.set_sojourn_time()
                self.next_event_time = self.curr_sojourn_time
        return arrivals