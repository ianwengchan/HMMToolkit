# input: data O, time T, state set S, edge set L, initial guess Q0
# output: Q



# calculate distinct_time_list
# calculate distinct_time_Pt_list

# find all distinct time intervals t_delta from T
# compute P(t_delta) = expm(Q * t_delta)

while ((iter < iter_max) & (ll_old - ll_new) > 1e-08)

    # compute posterior state probabilities
    # compute complete data loglikelihood using forward-backward or Viterbi

    # create soft count table C from posterior state probabilities

    # compute E_n_ij
    # compute E_tau_i

    q_ij = E_n_ij / E_tau_i

    q_ii = -sum(q_ij)

end