import torch

from hdm.dem.naive import internal_energy_static

from .internal_energy import internal_energy_dynamic_naive, internal_energy_dynamic_batched_manyhess

def free_action_naive(
        # how many terms are there in mu_x and mu_v?
        m_x, m_v,

        # how many derivatives are we tracking in generalised vectors?
        p,

        # dynamic terms
        mu_x_tildes, mu_v_tildes, # iterator of state and input mean estimates in generalized coordinates
        sig_x_tildes, sig_v_tildes, # as above but for covariance estimates
        y_tildes, # iterator of outputs in generalized coordinates
        eta_v_tildes, p_v_tildes, # iterator of input mean priors

        # prior means and precisions
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,

        # parameter estimate means and covariances
        mu_theta,
        mu_lambda,
        sig_theta,
        sig_lambda,

        # functions
        g, # output function
        f, # state transition function

        # noise precision matrices (to be scaled by hyperparameters)
        omega_w,
        omega_z,

        # generalized noise temporal autocorrelation inverse (precision)
        noise_autocorr_inv,
        ):

    u_c, u_c_theta_dd, u_c_lambda_dd = internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,
        compute_dds=True
    )
    f_c = u_c + (torch.logdet(sig_theta) + torch.logdet(sig_lambda)) / 2
    f_tsum = 0
    for t, (mu_x_tilde, mu_v_tilde,
            sig_x_tilde, sig_v_tilde,
            y_tilde,
            eta_v_tilde, p_v_tilde) in enumerate(
                zip(mu_x_tildes, mu_v_tildes,
                    sig_x_tildes, sig_v_tildes,
                    y_tildes,
                    eta_v_tildes, p_v_tildes,
                    strict=True)
            ):
        u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd = internal_energy_dynamic_naive(
            g, f, mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p, mu_theta, eta_v_tilde, p_v_tilde,
            mu_lambda, omega_w, omega_z, noise_autocorr_inv)

        # mean-field terms
        # FIXME OPT: Section 11.1 of Anil Meera & Wisse shows that gradients
        # along w_lambda are 0, so it might be unnecessary to compute.
        w_x_tilde, w_v_tilde, w_theta, w_lambda = [
            torch.trace(sig @ (u_c_dd + u_t_dd)) / 2
                for (sig, u_c_dd, u_t_dd) in [
                    (sig_x_tilde, 0, u_t_x_tilde_dd),
                    (sig_v_tilde, 0, u_t_v_tilde_dd),
                    (sig_theta, u_c_theta_dd, u_t_theta_dd),
                    (sig_lambda, u_c_lambda_dd, u_t_lambda_dd),
                ]
            ]
        f_tsum += u_t \
                + (torch.logdet(sig_x_tilde) + torch.logdet(sig_v_tilde)) / 2 \
                + w_x_tilde + w_v_tilde + w_theta + w_lambda
    f_bar = f_c + f_tsum
    return f_bar

def _batch_diag(tensor):
    # As mentioned here:
    # https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504/2
    return tensor.diagonal(offset=0, dim1=-1, dim2=-2)

def free_action_batched(
        # how many terms are there in mu_x and mu_v?
        m_x, m_v,

        # how many derivatives are we tracking in generalised vectors?
        p,

        # dynamic terms
        mu_x_tildes, mu_v_tildes, # iterator of state and input mean estimates in generalized coordinates
        sig_x_tildes, sig_v_tildes, # as above but for covariance estimates
        y_tildes, # iterator of outputs in generalized coordinates
        eta_v_tildes, p_v_tildes, # iterator of input mean priors

        # prior means and precisions
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,

        # parameter estimate means and covariances
        mu_theta,
        mu_lambda,
        sig_theta,
        sig_lambda,

        # functions
        g, # output function
        f, # state transition function

        # noise precision matrices (to be scaled by hyperparameters)
        omega_w,
        omega_z,

        # generalized noise temporal autocorrelation inverse (precision)
        noise_autocorr_inv,
        ):
    # Constant terms of free action
    u_c, u_c_theta_dd, u_c_lambda_dd = internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,
        compute_dds=True
    )
    f_c = u_c + (torch.logdet(sig_theta) + torch.logdet(sig_lambda)) / 2

    # Dynamic terms of free action that vary with time
    u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd = internal_energy_dynamic_batched_manyhess(
        g, f, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv)
    w_x_tilde_sum_ = _batch_diag(torch.bmm(sig_x_tildes, u_t_x_tilde_dd)).sum()
    w_v_tilde_sum_ = _batch_diag(torch.bmm(sig_v_tildes, u_t_v_tilde_dd)).sum()
    # w_theta and w_lambda are sums already, because u_t_theta_dd is a sum
    # because of how the batch Hessian is computed
    w_theta_sum_ = torch.trace(sig_theta @ (u_c_theta_dd + u_t_theta_dd))
    w_lambda_sum_ = torch.trace(sig_lambda @ (u_c_lambda_dd + u_t_lambda_dd))

    f_tsum = torch.sum(u_t) \
            + (torch.sum(torch.logdet(sig_x_tildes)) + torch.sum(torch.logdet(sig_v_tildes))) / 2 \
            + (w_x_tilde_sum_ + w_v_tilde_sum_ + w_theta_sum_ + w_lambda_sum_) / 2

    f_bar = f_c + f_tsum
    return f_bar
