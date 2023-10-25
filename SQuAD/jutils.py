import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None):
    """Generate samples from a correlated Gaussian distribution."""
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

    if cubic is not None:
        y = y ** 3

    return x, y


def rho_to_mi(dim, rho):
    return -0.5 * np.log(1-rho**2) * dim


def mi_to_rho(dim, mi):
    return np.sqrt(1-np.exp(-2.0 / dim * mi))


def mi_schedule(n_iter):
    """Generate schedule for increasing correlation over time."""
    mis = np.round(np.linspace(0.5, 5.5-1e-9, n_iter)) * 2.0
    return mis.astype(np.float32)


def logmeanexp_diag(x):
    batch_size = x.size(0)

    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).cuda()


def logmeanexp_nodiag(x, dim=None, device='cuda'):
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)

    logsumexp = torch.logsumexp(
        x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)

    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def tuba_lower_bound(scores, log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:, None]
    batch_size = scores.size(0)

    # First term is an expectation over samples from the joint,
    # which are the diagonal elmements of the scores matrix.
    joint_term = scores.diag().mean()

    # Second term is an expectation over samples from the marginal,
    # which are the off-diagonal elements of the scores matrix.
    marg_term = logmeanexp_nodiag(scores).exp()
    return 1. + joint_term - marg_term


def nwj_lower_bound(scores):
    return tuba_lower_bound(scores - 1.)


def infonce_lower_bound(scores):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    # Alternative implementation:
    # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
    mi = torch.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()
    return mi


def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


def js_lower_bound(f):
    nwj = nwj_lower_bound(f)
    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        nwj_js = nwj - js

    return js + nwj_js


def dv_upper_lower_bound(f):
    """DV lower bound, but upper bounded by using log outside."""
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    return first_term - second_term


def mine_lower_bound(f, buffer=None, momentum=0.9):
    if buffer is None:
        buffer = torch.tensor(1.0).cuda()
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new

    third_term_grad = buffer_update / buffer_new

    return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update


def regularized_dv_bound(f, l=0.0):
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    reg_term = l * (second_term.exp() - 1) ** 2

    with torch.no_grad():
        reg_term_no_grad = reg_term

    return first_term - second_term + reg_term - reg_term_no_grad


def renorm_q(f, alpha=1.0, clip=None):
    if clip is not None:
        f = torch.clamp(f * alpha, -clip, clip)
    z = logmeanexp_nodiag(f * alpha, dim=(0, 1))
    return z


def disc_renorm_q(f):
    batch_size = f.size(0)
    z = torch.zeros(1, requires_grad=True, device='cuda')

    opt = optim.SGD([z], lr=0.001)
    for i in range(10):
        opt.zero_grad()

        first_term = -F.softplus(z - f).diag().mean()
        st = -F.softplus(f - z)
        second_term = (st - st.diag().diag()).sum() / \
            (batch_size * (batch_size - 1.))
        total = first_term + second_term

        total.backward(retain_graph=True)
        opt.step()

        if total.item() <= -2 * np.log(2):
            break

    return z


def renorm_p(f, alpha=1.0):
    z = logmeanexp_diag(-f * alpha)
    return z


def smile_lower_bound(f, alpha=1.0, clip=None):
    z = renorm_q(f, alpha, clip)
    dv = f.diag().mean() - z

    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js


def js_dv_disc_renorm_lower_bound(f):
    z = disc_renorm_q(f)
    dv = f.diag().mean() - z.mean()

    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js


def vae_lower_bound(f):
    f1, f2 = f
    n = f1.size(0)
    logp = f1.mean()
    logq = (f2.sum() - f2.diag().sum()) / (n * (n-1.))

    with torch.no_grad():
        logq_nograd = logq * 1.0
        logqd = f2.diag().mean()
        logp_nograd = logp

    return logp - logqd + logq - logq_nograd  # logp - logqd + logq - logq_nograd


def js_nwj_renorm_lower_bound(f, alpha=1.0):
    z = renorm_q(f - 1.0, alpha)

    nwj = nwj_lower_bound(f - z)
    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        nwj_js = nwj - js

    return js + nwj_js


def estimate_p_norm(f, alpha=1.0):
    z = renorm_q(f, alpha)
    # f = renorm_p(f, alpha)
    # f = renorm_q(f, alpha)
    f = f - z
    f = -f

    return f.diag().exp().mean()


def estimate_mutual_information(estimator, x, y, critic_fn,
                                baseline_fn=None, alpha_logit=None, **kwargs):
    """Estimate variational lower bounds on mutual information.

  Args:
    estimator: string specifying estimator, one of:
      'nwj', 'infonce', 'tuba', 'js', 'interpolated'
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix
    baseline_fn (optional): callable that takes y as input 
      outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound

  Returns:
    scalar estimate of mutual information
    """
    x, y = x.cuda(), y.cuda()
    scores = critic_fn(x, y)
    if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))
    if estimator == 'infonce':
        mi = infonce_lower_bound(scores)
    elif estimator == 'nwj':
        mi = nwj_lower_bound(scores)
    elif estimator == 'tuba':
        mi = tuba_lower_bound(scores, log_baseline)
    elif estimator == 'js':
        mi = js_lower_bound(scores)
    elif estimator == 'smile':
        mi = smile_lower_bound(scores, **kwargs)
    elif estimator == 'dv_disc_normalized':
        mi = js_dv_disc_renorm_lower_bound(scores, **kwargs)
    elif estimator == 'nwj_normalized':
        mi = js_nwj_renorm_lower_bound(scores, **kwargs)
    elif estimator == 'dv':
        mi = dv_upper_lower_bound(scores)
    # p_norm = estimate_p_norm(scores * kwargs.get('alpha', 1.0))
    if estimator is not 'smile':
        p_norm = renorm_q(scores)
    else:
        p_norm = renorm_q(scores, alpha=kwargs.get(
            'alpha', 1.0), clip=kwargs.get('clip', None))
    return mi, p_norm


def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


class SeparableCritic(nn.Module):
    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores


class ConcatCritic(nn.Module):
    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()


def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)


dim = 20


CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic,
}

BASELINES = {
    'constant': lambda: None,
    'unnormalized': lambda: mlp(dim=dim, hidden_dim=512, output_dim=1, layers=2, activation='relu').cuda(),
    'gaussian': lambda: log_prob_gaussian,
}


def train_estimator(critic_params, data_params, mi_params, opt_params, **kwargs):
    """Main training loop that estimates time-varying MI."""
    # Ground truth rho is only used by conditional critic
    critic = CRITICS[mi_params.get('critic', 'separable')](
        rho=None, **critic_params).cuda()
    baseline = BASELINES[mi_params.get('baseline', 'constant')]()

    opt_crit = optim.Adam(critic.parameters(), lr=opt_params['learning_rate'])
    if isinstance(baseline, nn.Module):
        opt_base = optim.Adam(baseline.parameters(),
                              lr=opt_params['learning_rate'])
    else:
        opt_base = None

    def train_step(rho, data_params, mi_params):
        # Annoying special case:
        # For the true conditional, the critic depends on the true correlation rho,
        # so we rebuild the critic at each iteration.
        opt_crit.zero_grad()
        if isinstance(baseline, nn.Module):
            opt_base.zero_grad()

        if mi_params['critic'] == 'conditional':
            critic_ = CRITICS['conditional'](rho=rho).cuda()
        else:
            critic_ = critic

        x, y = sample_correlated_gaussian(
            dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'], cubic=data_params['cubic'])
        mi, p_norm = estimate_mutual_information(
            mi_params['estimator'], x, y, critic_, baseline, mi_params.get('alpha_logit', None), **kwargs)
        loss = -mi

        loss.backward()
        opt_crit.step()
        if isinstance(baseline, nn.Module):
            opt_base.step()

        return mi, p_norm

    # Schedule of correlation over iterations
    mis = mi_schedule(opt_params['iterations'])
    rhos = mi_to_rho(data_params['dim'], mis)

    estimates = []
    p_norms = []
    for i in range(opt_params['iterations']):
        mi, p_norm = train_step(
            rhos[i], data_params, mi_params)
        mi = mi.detach().cpu().numpy()
        p_norm = p_norm.detach().cpu().numpy()
        estimates.append(mi)
        p_norms.append(p_norm)

    return np.array(estimates), np.array(p_norms)
