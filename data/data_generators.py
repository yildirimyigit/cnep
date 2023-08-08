import torch

def generate_cx_sigm(x:torch.Tensor):
    # rand_btw_7-30 * {-1, 1}
    c1 = torch.rand_like(x) * 1 + 7 #* torch.from_numpy(np.random.choice([-1, 1], (n, 1)))  # c1 for steepness
    c2 = 0.5  # c2 midpoint

    noise = torch.clamp(torch.randn_like(x)*1e-4**0.5, min=0)
    data = 1/(1 + torch.exp(-c1 * (x-c2))) + noise
    return data


def generate_reverse_cx_sigm(x:torch.Tensor):
    return 1-generate_cx_sigm(x)


def generate_sin(x:torch.Tensor):
    dx = x.shape[-1]
    noise = torch.clamp(torch.randn_like(x)*1e-4**0.5, min=0).view(-1, dx)
    noise = 0   # TODO: remove this
    return torch.sin(x*torch.pi) + noise


def generate_cos(x:torch.Tensor):
    dx = x.shape[-1]
    noise = torch.clamp(torch.randn_like(x)*1e-4**0.5, min=0).view(-1, dx)
    return torch.cos((0.5+x)*torch.pi) + 1 + noise

def generate_chainsaw(x:torch.Tensor, amp=1, mod=0.5, eps=0.1):
    r1 = torch.rand(1)*(eps*2)+(1-eps)
    r2 = torch.rand(1)*(eps*2)+(1-eps)
    r3 = torch.rand(1)*(eps*2)+(1-eps)
    return r1*amp*((x+r2) % (mod))+r3-amp/4


def generate_reverse_chainsaw(x:torch.Tensor, amp=1, mod=0.5, eps=0.1):
    r1 = torch.rand(1)*(eps*2)+(1-eps)
    r2 = torch.rand(1)*(eps*2)+(1-eps)
    r3 = torch.rand(1)*(eps*2)+(1-eps)
    return -r1*amp*((x+r2) % (mod))+r3+amp/4

def generate_gaussian(x:torch.Tensor, mu, sigma, noisy=True):
    y = torch.exp(torch.distributions.Normal(mu, sigma).log_prob(x))
    dx = x.shape[-1]
    noise = torch.clamp(torch.randn_like(x)*1e-7**0.5, min=0).view(-1, dx)
    return y + noise if noisy else y

def generate_reverse_gaussian(x:torch.Tensor, mu, sigma, noisy=True):
    y = torch.exp(torch.distributions.Normal(mu, sigma).log_prob(x))
    dx = x.shape[-1]
    noise = torch.clamp(torch.randn_like(x)*1e-7**0.5, min=0).view(-1, dx)
    return -1 * (y + noise if noisy else y)