import torch

def generate_cx_sigm(x):
    # rand_btw_7-30 * {-1, 1}
    c1 = torch.rand(len(x)) * 1 + 7 #* torch.from_numpy(np.random.choice([-1, 1], (n, 1)))  # c1 for steepness
    c2 = 0.5  # c2 midpoint
    
    data = 1/(1 + torch.exp(-c1 * (x-c2))) + torch.clamp(torch.randn(len(x))*1e-2**0.5, min=0)
    return data


def generate_reverse_cx_sigm(x):
    return 1-generate_cx_sigm(x)


def generate_sin(x, amp=1, omega=1, eps=0.1):
    noise = torch.clamp(torch.randn(len(x))*1e-2**0.5, min=0)
    return torch.sin(x*torch.pi) + noise


def generate_cos(x, amp=1, omega=1, eps=0.1):
    noise = torch.clamp(torch.randn(len(x))*1e-2**0.5, min=0)
    return torch.cos((0.5+x)*torch.pi) + 1 + noise

def generate_chainsaw(x, amp=1, mod=0.5, eps=0.1):
    r1 = torch.rand(1)*(eps*2)+(1-eps)
    r2 = torch.rand(1)*(eps*2)+(1-eps)
    r3 = torch.rand(1)*(eps*2)+(1-eps)
    return r1*amp*((x+r2) % (mod))+r3-amp/4


def generate_reverse_chainsaw(x, amp=1, mod=0.5, eps=0.1):
    r1 = torch.rand(1)*(eps*2)+(1-eps)
    r2 = torch.rand(1)*(eps*2)+(1-eps)
    r3 = torch.rand(1)*(eps*2)+(1-eps)
    return -r1*amp*((x+r2) % (mod))+r3+amp/4