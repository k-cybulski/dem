
from hdm.core import iterate_generalized, deriv_mat
from hdm.dem.naive import kron
import torch

from .util import _fix_grad_shape, _fix_grad_shape_batch
from .generalized_func import generalized_func_batched
from .generalized_func import generalized_func_naive as generalized_func

def internal_energy_dynamic_naive(
        g, f, mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p, mu_theta, eta_v_tilde, p_v_tilde,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv):
    """
    Computes dynamic terms of the internal energy for a single timestep, along
    with necessary Hessians. These are the precision-weighted errors and
    precision log determinants on the dynamic states. Hessians are returned for
    parameters theta and hyperparameters lambda as well.
    """
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)
    # make a temporary function which we can use to compute hessians w.r.t. the relevant parameters
    # for the computation of mean-field terms
    def _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda):
        g_tilde = generalized_func(g, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)
        f_tilde = generalized_func(f, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)

        err_y = y_tilde - g_tilde
        err_v = mu_v_tilde - eta_v_tilde
        err_x = deriv_mat_x @ mu_x_tilde - f_tilde

        # we need to split up mu_lambda into the hyperparameter for noise of states and of outputs
        # the hyperparameters are just a single lambda scalar, one for the states and one for the outputs
        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -err_y.T @ prec_z_tilde @ err_y + torch.logdet(prec_z_tilde)
        u_t_v_ = -err_v.T @ p_v_tilde @ err_v + torch.logdet(p_v_tilde)
        u_t_x_ = -err_x.T @ prec_w_tilde @ err_x + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t
    u_t = _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda)
    # horribly inefficient way to go about this, but hey, at least it may work...
    # (so many unnecessary repeated computations)

    # FIXME OPT: Optimize the code below. Don't run
    # torch.autograd.functional.hessian four times separately? Running it once
    # should allow for all the necessary outputs. But it might unnecessarily
    # compute Hessians _between_ the parameters, which might be slower?
    u_t_x_tilde_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu, mu_v_tilde, mu_theta, mu_lambda), mu_x_tilde, create_graph=True)
    u_t_v_tilde_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu, mu_theta, mu_lambda), mu_v_tilde, create_graph=True)
    u_t_theta_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu, mu_lambda), mu_theta, create_graph=True)
    u_t_lambda_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu), mu_lambda, create_graph=True)

    u_t_x_tilde_dd = _fix_grad_shape(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd )
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)
    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd


def internal_energy_dynamic_onehess(
        g, f, mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p, mu_theta, eta_v_tilde, p_v_tilde,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv):
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)
    # make a temporary function which we can use to compute hessians w.r.t. the relevant parameters
    # for the computation of mean-field terms
    def _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda):
        g_tilde = generalized_func(g, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)
        f_tilde = generalized_func(f, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)

        err_y = y_tilde - g_tilde
        err_v = mu_v_tilde - eta_v_tilde
        err_x = deriv_mat_x @ mu_x_tilde - f_tilde

        # we need to split up mu_lambda into the hyperparameter for noise of states and of outputs
        # the hyperparameters are just a single lambda scalar, one for the states and one for the outputs
        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -err_y.T @ prec_z_tilde @ err_y + torch.logdet(prec_z_tilde)
        u_t_v_ = -err_v.T @ p_v_tilde @ err_v + torch.logdet(p_v_tilde)
        u_t_x_ = -err_x.T @ prec_w_tilde @ err_x + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t
    u_t = _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda)
    # horribly inefficient way to go about this, but hey, at least it may work...
    # (so many unnecessary repeated computations)

    # FIXME OPT: Optimize the code below. Don't run
    # torch.autograd.functional.hessian four times separately? Running it once
    # should allow for all the necessary outputs. But it might unnecessarily
    # compute Hessians _between_ the parameters, which might be slower?
    dds = torch.autograd.functional.hessian(
            lambda mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda: _int_eng_dynamic(
                mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda),
            (mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda),
            create_graph=True)
    u_t_x_tilde_dd = dds[0][0]
    u_t_v_tilde_dd = dds[1][1]
    u_t_theta_dd = dds[2][2]
    u_t_lambda_dd = dds[3][3]

    u_t_x_tilde_dd = _fix_grad_shape(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd )
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)
    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd


def internal_energy_dynamic_batched(
        g, f, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv):
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)

    def _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda):
        mu_theta = mu_theta
        mu_lambda = mu_lambda
        g_tildes = generalized_func_batched(g, mu_x_tildes, mu_v_tildes, m_x, m_v, p, mu_theta)
        f_tildes = generalized_func_batched(f, mu_x_tildes, mu_v_tildes, m_x, m_v, p, mu_theta)

        err_y = y_tildes - g_tildes
        err_v = mu_v_tildes - eta_v_tildes
        err_x = torch.matmul(deriv_mat_x, mu_x_tildes) - f_tildes

        n_batch = mu_x_tildes.shape[0]

        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -torch.bmm(err_y.mT, torch.matmul(prec_z_tilde, err_y)).reshape(n_batch) + torch.logdet(prec_z_tilde)
        u_t_v_ = -torch.bmm(err_v.mT, torch.bmm(p_v_tildes, err_v)).reshape(n_batch) + torch.logdet(p_v_tildes)
        u_t_x_ = -torch.bmm(err_x.mT, torch.matmul(prec_w_tilde, err_x)).reshape(n_batch) + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t

    u_t = _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)
    dds = torch.autograd.functional.hessian(
            lambda mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda: torch.sum(_int_eng_dynamic(
                mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)),
            (mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda),
            create_graph=True)
    u_t_x_tilde_dd = dds[0][0]
    u_t_v_tilde_dd = dds[1][1]
    u_t_theta_dd = dds[2][2]
    u_t_lambda_dd = dds[3][3]

    u_t_x_tilde_dd = _fix_grad_shape_batch(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape_batch(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd)
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)
    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd


def internal_energy_dynamic_batched_manyhess(
        g, f, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv):
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)

    def _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda):
        mu_theta = mu_theta
        mu_lambda = mu_lambda
        g_tildes = generalized_func_batched(g, mu_x_tildes, mu_v_tildes, m_x, m_v, p, mu_theta)
        f_tildes = generalized_func_batched(f, mu_x_tildes, mu_v_tildes, m_x, m_v, p, mu_theta)

        err_y = y_tildes - g_tildes
        err_v = mu_v_tildes - eta_v_tildes
        err_x = torch.matmul(deriv_mat_x, mu_x_tildes) - f_tildes

        n_batch = mu_x_tildes.shape[0]

        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -torch.bmm(err_y.mT, torch.matmul(prec_z_tilde, err_y)).reshape(n_batch) + torch.logdet(prec_z_tilde)
        u_t_v_ = -torch.bmm(err_v.mT, torch.bmm(p_v_tildes, err_v)).reshape(n_batch) + torch.logdet(p_v_tildes)
        u_t_x_ = -torch.bmm(err_x.mT, torch.matmul(prec_w_tilde, err_x)).reshape(n_batch) + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t

    u_t = _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)
    u_t_x_tilde_dd = torch.autograd.functional.hessian(lambda mu: torch.sum(_int_eng_dynamic(mu, mu_v_tildes, mu_theta, mu_lambda)), mu_x_tildes, create_graph=True)
    u_t_v_tilde_dd = torch.autograd.functional.hessian(lambda mu: torch.sum(_int_eng_dynamic(mu_x_tildes, mu, mu_theta, mu_lambda)), mu_v_tildes, create_graph=True)
    u_t_theta_dd = torch.autograd.functional.hessian(lambda mu: torch.sum(_int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu, mu_lambda)), mu_theta, create_graph=True)
    u_t_lambda_dd = torch.autograd.functional.hessian(lambda mu: torch.sum(_int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu)), mu_lambda, create_graph=True)

    u_t_x_tilde_dd = _fix_grad_shape_batch(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape_batch(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd)
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)
    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd
