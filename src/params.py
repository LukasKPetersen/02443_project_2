NUM_SOURCES = 4
NUM_SERVERS = 1
LAMBDA_ON = 2  # arrival rate when the source is ON
OMEGA_ON = 0.8  # state change rate from ON to OFF
OMEGA_OFF = 0.2  # state change rate from OFF to ON
ARRIVAL_TO_SERVICE_RATE_RATIO = 10/9  # ratio of arrival rate to service rate


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