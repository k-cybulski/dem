
from hdm.core import deriv_mat
from hdm.dem.naive import kron
import torch

from .util import _fix_grad_shape, _fix_grad_shape_batch
from .generalized_func import generalized_func_batched
from .generalized_func import generalized_func_naive as generalized_func

### First implementation

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

### Implementation below runs torch.autograd.functional.hessian just once instead of four times

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

### Implementation below is batched and runs torch.autograd.functional.hessian just once

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

### Implementation below is batched and runs torch.autograd.functional.hessian multiple times
### Conclusion: It's the quickest so far

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


##
## Avoiding unnecessary gradient applications
##
### The implementations below try to avoid calling autograd functions multiple
### times when unnecessary, and instead do it just once
### Conclusion: It's slower than just to run torch.autograd.functional.hessian multiple times

def output_sum_and_dds(func, *args):
    args_rg = [
            arg.detach().requires_grad_()
            for arg in args
        ]
    appl = torch.sum(func(*args_rg))
    args_ds = torch.autograd.grad(appl, args_rg, create_graph=True)
    args_dds = []
    for arg_rg, arg_d in zip(args_rg, args_ds):
        if arg_d.ndim == 3: # it's a batch
            arg_dd = []
            for bnum, arg_d_b in enumerate(arg_d):
                arg_dd_b = torch.stack([
                    torch.autograd.grad(d, arg_rg, create_graph=True)[0][bnum]
                    for d in arg_d_b
                ])
                # NOTE: The last dimension is here because our dynamic vectors
                # are of shape (-1, 1)
                arg_dd_b = arg_dd_b[:, :, 0]
                arg_dd.append(arg_dd_b)
            arg_dd = torch.stack(arg_dd)
        else:
            arg_dd = torch.stack([
                torch.autograd.grad(d, arg_rg, create_graph=True)[0]
                for d in arg_d
            ])
        args_dds.append(arg_dd)
    return appl, args_ds, args_dds

def internal_energy_dynamic_batched_customhess(
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

    u_t, _, (u_t_x_tilde_dd, u_t_v_tilde_dd,
             u_t_theta_dd, u_t_lambda_dd) = output_sum_and_dds(_int_eng_dynamic, mu_x_tildes,
                                                               mu_v_tildes, mu_theta, mu_lambda)

    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd

### The implementation below avoids computing a Hessian with terms between-batches, since these are always zero
### Conclusion: It's slower to do these gymnastics than to just accept and ignore these zero-terms

def _prep_for_grad(tensor):
    # If we are working on batches, then we want each batch to be treated separately w.r.t. gradient computations
    if tensor.ndim == 3: # has batches
        return tensor.shape[0], [
            batch.detach().requires_grad_()
            for batch in tensor
        ]
    else:
        return 0, tensor.detach().requires_grad_()


def output_sum_and_dds_nocrossbatch(func, *args):
    args_rg = [
            _prep_for_grad(arg)
            for arg in args
        ]
    args_batched = [
            arg if not(isinstance(arg, list)) else torch.stack(arg)
            for batch_num, arg in args_rg
        ]
    appl = torch.sum(func(*args_batched))
    # Jacobians
    args_longlist = []
    for batch_num, arg in args_rg:
        if batch_num == 0:
            args_longlist.append(arg)
        else:
            args_longlist.extend(arg)
    args_ds_longlist = torch.autograd.grad(appl, args_longlist, create_graph=True)
    args_ds = []
    cursor = 0
    for batch_num, _ in args_rg:
        if batch_num == 0:
            args_ds.append(args_ds_longlist[cursor])
            cursor += 1
        else:
            args_ds.append(torch.stack(args_ds_longlist[cursor:(cursor+batch_num)]))
            cursor += batch_num
    # Hessians
    args_dds_longlist = []
    for arg_rg, arg_d in zip(args_longlist, args_ds_longlist):
        arg_dd = torch.stack([
            torch.autograd.grad(d, arg_rg, create_graph=True)[0]
            for d in arg_d
        ])
        if arg_dd.shape[-1] == 1:
            arg_dd = arg_dd[..., 0]
        args_dds_longlist.append(arg_dd)

    args_dds = []
    cursor = 0
    for batch_num, _ in args_rg:
        if batch_num == 0:
            args_dds.append(args_dds_longlist[cursor])
            cursor += 1
        else:
            args_dds.append(torch.stack(args_dds_longlist[cursor:(cursor+batch_num)]))
            cursor += batch_num

    return appl, args_ds, args_dds

def internal_energy_dynamic_batched_customhess_nocrossbatch(
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

    u_t, _, (u_t_x_tilde_dd, u_t_v_tilde_dd,
             u_t_theta_dd, u_t_lambda_dd) = output_sum_and_dds_nocrossbatch(_int_eng_dynamic,
                                                                            mu_x_tildes,
                                                                            mu_v_tildes,
                                                                            mu_theta, mu_lambda)

    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd
