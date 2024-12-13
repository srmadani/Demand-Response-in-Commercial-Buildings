from d3qn import main as main_d3qn
from sacd import main as main_sacd
from ppod import main as main_ppod
from ddpg import main as main_ddpg
from ppo  import main as main_ppo
from td3  import main as main_td3
import pickle
import numpy as np

def results_agg(hvac_alg = 'ppod', ctrl_alg = 'td3', cpp=0.000067, temperature=0):
    hvac_alg = hvac_alg.lower()
    ctrl_alg = ctrl_alg.lower()
    C_values = [100, 150]
    R_values = [1, 2]
    h_values = [50, 80]
    # alpha_values = [0.1, 0.2, 0.3, 0.4]
    # beta_values  = [0.1, 0.2, 0.3, 0.4]
    alpha_values = [0.2, 0.3]
    beta_values  = [0.2, 0.3]
    # gamma_values = [0.35132, 0.45132, 0.65132, 0.75132]
    gamma = 0.75132
    main_res            = {}
    agg_load_total          = np.zeros(7*24*12)
    agg_load_hvac           = np.zeros(7*24*12)
    agg_load_ctrl           = np.zeros(7*24*12)
    agg_main_Tin            = np.zeros(7*24*12)
    agg_ref_total_load      = np.zeros(7*24*12)
    agg_ref_hvac_load       = np.zeros(7*24*12)
    agg_ref_ctrl_load       = np.zeros(7*24*12)
    agg_ref_Tin             = np.zeros(7*24*12)
    agg_total_nonreducible  = np.zeros(7*24*12)
    agg_cost_components     = np.zeros((7*24*12, 4))
    agg_total_cost          = 0
    
    # Initialize the main_res dictionary with default values
    for C in C_values:
        for R in R_values:
            for h in h_values:
                for alpha in alpha_values:
                    for beta in beta_values:
                        params = (C, R, h, alpha, beta)
                        main_res[params] = {
                            'score': 0,
                            'load_total': [],
                            'load_hvac': [],
                            'load_ctrl': [],
                            'cost': 0,
                            'T_in': []
                        }
    
    with open(r'datasets/ref_res.pkl', 'rb') as f:
        ref_res = pickle.load(f)
    with open(r'datasets/df.pkl', 'rb') as f:
        db = pickle.load(f)
    db = db.iloc[(8-1) * 7 * 12 * 24 : 8 * 7 * 12 * 24,:].copy()
    for beta in beta_values:
        if ctrl_alg == 'ddpg':
            score_, load_ctrl, cost_, cost_components_ = main_ddpg(beta=beta, render=True, compare=False, gamma=gamma)
        elif ctrl_alg == 'td3':
            score_, load_ctrl, cost_, cost_components_ = main_td3(beta=beta, render=True, compare=False, gamma=gamma)
        else:
            score_, load_ctrl, cost_, cost_components_ = main_ppo(beta=beta, render=True, compare=False, gamma=gamma)
        for C in C_values:
            for R in R_values:
                for h in h_values:
                    ref_hvac = []
                    ref_T_in = []
                    for day in range((8-1)*7+1,8*7+1):
                        ref_hvac.extend(ref_res[(day, C, R, h)][0])
                        ref_T_in.extend(ref_res[(day, C, R, h)][1])
                    for alpha in alpha_values:
                        if hvac_alg == 'd3qn':
                            score, T_in, load_hvac, cost, cost_components = main_d3qn(C=C, R=R, h=h, alpha=alpha, render=True, gamma=gamma, temperature=temperature)
                        elif hvac_alg == 'ppod':
                            score, T_in, load_hvac, cost, cost_components = main_ppod(C=C, R=R, h=h, alpha=alpha, render=True, gamma=gamma, temperature=temperature)
                        else:
                            score, T_in, load_hvac, cost, cost_components = main_sacd(C=C, R=R, h=h, alpha=alpha, render=True, gamma=gamma, temperature=temperature)
                        params = (C, R, h, alpha, beta)
                        main_res[params]['cost'] = float(cost) + float(cost_) + np.sum(db['non_reducible [kWh]'].to_numpy() * db['P [$/kWh]'])
                        main_res[params]['score'] = score + score_
                        main_res[params]['load_total'] = load_hvac + load_ctrl + db['non_reducible [kWh]'].to_numpy()
                        main_res[params]['load_hvac'] = load_hvac
                        main_res[params]['load_ctrl'] = load_ctrl
                        main_res[params]['T_in'] = T_in
                        main_res[params]['cost_components'] = cost_components + cost_components_

                        agg_total_cost          += main_res[params]['cost']
                        agg_load_total          += main_res[params]['load_total']
                        agg_load_hvac           += load_hvac
                        agg_load_ctrl           += load_ctrl
                        agg_main_Tin            += T_in
                        agg_ref_total_load      += np.array(ref_hvac) * h + db['reducible [kWh]'].to_numpy() + db['non_reducible [kWh]'].to_numpy()
                        agg_ref_hvac_load       += np.array(ref_hvac) * h
                        agg_ref_ctrl_load       += db['reducible [kWh]'].to_numpy()
                        agg_ref_Tin             += np.array(ref_T_in)
                        agg_total_nonreducible  += db['non_reducible [kWh]'].to_numpy()
                        agg_cost_components     +=  cost_components + cost_components_

    agg_res = {}

    agg_res['agg_total_cost'] = agg_total_cost
    agg_res['agg_load_total'] = agg_load_total
    agg_res['agg_load_hvac'] = agg_load_hvac
    agg_res['agg_load_ctrl'] = agg_load_ctrl
    agg_res['agg_main_Tin'] = agg_main_Tin/32
    agg_res['agg_ref_total_load'] = agg_ref_total_load
    agg_res['agg_ref_hvac_load'] = agg_ref_hvac_load
    agg_res['agg_ref_ctrl_load'] = agg_ref_ctrl_load
    agg_res['agg_ref_Tin'] = agg_ref_Tin/32
    agg_res['agg_total_nonreducible'] = agg_total_nonreducible
    agg_res['agg_cost_components'] = agg_cost_components
    agg_res['db'] = db


    
    with open(f'res/main_res.pkl', 'wb') as file:
        pickle.dump(main_res, file)
    with open(f'res/agg_res.pkl', 'wb') as file:
        pickle.dump(agg_res, file)


def alg_comparison():
    C_values = [100, 150]
    R_values = [1, 2]
    h_values = [50, 80]
    alpha_values = [0.2, 0.3]
    beta_values = [0.2, 0.3]
    scores_hvac = np.zeros(3)
    scores_ctrl = np.zeros(3)
    for C in C_values:
        for R in R_values:
            for h in h_values:
                for alpha in alpha_values:
                    hvac_algs = [main_d3qn, main_ppod, main_sacd]
                    for cc, alg in enumerate(hvac_algs):
                        scores_hvac[cc] += alg(C=C, R=R, h=h, alpha=alpha, render=False, compare=True)
    for beta in beta_values:
        ctrl_algs = [main_ddpg, main_td3, main_ppo]
        for cc, alg in enumerate(ctrl_algs):
            scores_ctrl[cc] += alg(beta=beta, render=False, compare=True)

    print(f'total scores for hvac algorithms are: {scores_hvac}')
    print(f'best hvac algorithm is {scores_hvac.argmax()}')
    print(f'total scores for ctrl algorithms are: {scores_ctrl}')
    print(f'best ctrl algorithm is {scores_ctrl.argmax()}')


# total scores for hvac algorithms are: [-18360.95830439  -5884.05223127  -7219.71741781]
# best hvac algorithm is 1
# total scores for ctrl algorithms are: [-64.64607239 -64.41662598 -64.73800659]
# best ctrl algorithm is 1

if __name__ == '__main__':
    results_agg(hvac_alg = 'ppod', ctrl_alg = 'td3')
    # alg_comparison()
