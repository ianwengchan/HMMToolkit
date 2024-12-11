# Some helper functions for fitting zero-inflated distributions during EM algorithm

function EM_E_z_zero_obs_update(lower, prob, ll_vec)
    return lower == 0.0 ? prob / (prob + (1 - prob) * exp(ll_vec)) : 0.0
end

function EM_E_z_zero_obs(yl, p_old, gate_expert_ll_pos_comp)
    return EM_E_z_zero_obs_update.(yl, p_old, gate_expert_ll_pos_comp)
end

# EM

function EM_M_zero(z_zero_e_obs, z_pos_e_obs, z_zero_e_lat, z_pos_e_lat, k_e)
    num = sum(z_zero_e_obs .+ (z_zero_e_lat .* k_e))
    denom = num + sum(z_pos_e_obs .+ (z_pos_e_lat .* k_e))
    return num / denom
end