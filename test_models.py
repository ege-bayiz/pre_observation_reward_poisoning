import attacker
import environment as env
import victims
import bandit_algorithms as alg
import numpy as np

if __name__ == '__main__':
    environment = env.generate_10_arm_testbed('StandardNormal')

    victim_exp3 = victims.BanditVictim(alg.Exp3Algorithm(10, 0.001),
                                       environment,
                                       victims.ObservationModel.ACTION_REWARD,
                                       victims.AttackModel.CHOSEN_ARM,
                                       np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))

    victim_ucb = victims.BanditVictim(alg.UcbAlgorithm(10),
                                      environment,
                                      victims.ObservationModel.ACTION_REWARD,
                                      victims.AttackModel.CHOSEN_ARM,
                                      np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))

    ## Uncomment run all tests
    attacker.test_attackers(victim_exp3, "exp3", 1)
    attacker.test_attackers(victim_exp3, "exp3", 0.99)
    attacker.test_attackers(victim_ucb, "ucb", 1)
    attacker.test_attackers(victim_ucb, "ucb", 0.99)