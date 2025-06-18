import matplotlib.pyplot as plt
from IPP_simple import SingleServerIPP

intensity = 0.1  # intensity of arrivals when ON
gamma1 = 0.5  # rate of state transition from ON to OFF
gamma2 = 0.1  # rate of state transition from OFF to ON
runtime = 1000  # total simulation time

simulator = SingleServerIPP(lambda_on=intensity, omega1=gamma1, omega2=gamma2)
arrivals = simulator.run(run_time=runtime)

# some plots
plt.figure(figsize=(10, 5))
plt.hist(arrivals, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Arrivals in Single Server IPP Simulation')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.xlim(0, runtime)
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(arrivals, range(len(arrivals)), marker='o', linestyle='-', color='blue', alpha=0.7)
plt.title('Arrival Times in Single Server IPP Simulation')
plt.xlabel('Time')
plt.ylabel('Arrival Index')
plt.xlim(0, runtime)
plt.grid()
plt.show()