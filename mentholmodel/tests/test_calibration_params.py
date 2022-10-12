import numpy as np


def adjust_transition_probs_according_to_initiation_cessation_params(initiation_rate_decrease, continuation_rate_decrease, in_probs2345, in_probs1):

    """
    Tune probabilities according to initiation_rate_decrease
    and cessation_rate_factor parameters. 
    
    The initiation_rate_decrease param tells you by
    how much to deacrease the initiation rate, that is,
    1 - (probability of a never smoker staying a never smoker).

    The cessation_factor param tells you by how much to multiply
    the cessation probability, i.e. the probability that people 
    transition into group 2 (former smokers). "Reducing continuation"
    is semantically equivalent to "increasing cessation."

    Ex. If the probability of a person making the transition 1->1
    is .8, and we decrease the initiation rate by 30%, then the
    new probability of that person making the 1->1 is .86. 

    Ex. If the probability of a person making the transition 3->2
    has probability 0.1 and the continuation rate is decreased by 30%
    then the new probability of making the transition 3->2 is .37.
    """
    
    if initiation_rate_decrease > 0:
        in_probs1[:,0] += (1 - in_probs1[:,0]) * initiation_rate_decrease
        in_probs1[:,1:] -= in_probs1[:,1:] * initiation_rate_decrease

    if continuation_rate_decrease > 0:
        in_probs2345[:,1] += (1 - in_probs2345[:,1]) * continuation_rate_decrease
        in_probs2345[:,2:] -= in_probs2345[:,2:] * continuation_rate_decrease

    return in_probs2345, in_probs1


def print_probs(probs):
    s = np.sum(probs,axis=0)
    print(s / np.sum(s))

p2345 = np.random.rand(100,5)
p2345[:,0] = np.zeros_like(p2345[:,0])
p2345 /= np.sum(p2345, axis=1)[:,np.newaxis]
p1 = np.random.rand(100,5)
p1 /= np.sum(p1, axis=1)[:,np.newaxis]

print_probs(p2345)
print_probs(p1)

p2345, p1 = adjust_transition_probs_according_to_initiation_cessation_params(0.2, 0.2, p2345, p1)

print_probs(p2345)
print_probs(p1)