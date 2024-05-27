from concurrent.futures import ProcessPoolExecutor, as_completed
from d3qn import main as main_d3qn
from sacd import main as main_sacd
from ppod import main as main_ppod
from ddpg import main as ddpg_main
from ppo import main as ppo_main
from td3 import main as td3_main

def run_experiment(module_main, **kwargs):
    print(f"Running experiment using {module_main.__module__} with parameters: {kwargs}")
    module_main(**kwargs)

def run_experiments_in_parallel():
    with ProcessPoolExecutor() as executor:
        experiments = []
        # # D3QN, SACD, PPOD experiments
        for main_func in [main_d3qn, main_sacd, main_ppod]:
        # for main_func in [main_d3qn]:
            for C in C_values:
                for R in R_values:
                    for h in h_values:
                        for alpha in alpha_values:
                            experiments.append(executor.submit(run_experiment, main_func, C=C, R=R, h=h, alpha=alpha))

        DDPG, PPO, TD3 experiments
        for main_func in [ddpg_main, ppo_main, td3_main]:
        # for main_func in [td3_main]:
            for beta in beta_values:
                experiments.append(executor.submit(run_experiment, main_func, beta=beta))

        # Wait for all experiments to complete
        for future in as_completed(experiments):
            future.result()  # Handle exceptions here if needed

if __name__ == '__main__':
    C_values = [100, 150]
    R_values = [1, 2]
    h_values = [50, 80]
    # alpha_values = [0.1, 0.4]
    alpha_values = [0.1, 0.2, 0.3, 0.4]
    beta_values = [0.1, 0.2, 0.3, 0.4]  # Added beta_values list for DDPG, PPO, TD3

    run_experiments_in_parallel()