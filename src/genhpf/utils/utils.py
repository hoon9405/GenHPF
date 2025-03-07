import torch
import torch.nn.functional as F

def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

def sample_word(logits, strategy='topp', **kwargs):
    if strategy == 'topp':
        return sample_top_p(logits, **kwargs)
    else:
        raise NotImplementedError

def sample_top_p(logits, k=5, sample=False, **kwargs):
    if sample:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("Inf")    
        probs = F.softmax(out, dim=-1)

        ix = torch.multinomial(probs.squeeze(0), num_samples=1)
    else:
        _, ix = torch.topk(logits, k=1, dim=-1)
    
    next_words = ix

    return next_words

def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor

def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}
    
    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    
    return _apply(sample)

def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)

def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()
    return apply_to_sample(_move_to_cpu, sample)


import logging
from contextlib import contextmanager

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor

def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state

def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])

class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)

def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}
    
    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x
    
    return _apply(sample)

def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        return tensor.to(device=device, non_blocking=True)
    
    return apply_to_sample(_move_to_cuda, sample)

def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()
    return apply_to_sample(_move_to_cpu, sample)

def prepare_sample(sample):
    if torch.cuda.is_available():
        sample = move_to_cuda(sample)
    
    return sample

class CudaEnvironment(object):
    def __init__(self):
        cur_device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
        self.name = prop.name
        self.major = prop.major
        self.minor = prop.minor
        self.total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024

    @staticmethod
    def pretty_print_cuda_env_list(cuda_env_list):
        """
        Given a list of CudaEnviorments, pretty print them
        """
        num_workers = len(cuda_env_list)
        center = "CUDA enviroments for all {} workers".format(num_workers)
        banner_len = 40 - len(center) // 2
        first_line = "*" * banner_len + center + "*" * banner_len
        logger.info(first_line)
        for r, env in enumerate(cuda_env_list):
            logger.info(
                "rank {:3d}: ".format(r)
                + "capabilities = {:2d}.{:<2d} ; ".format(env.major, env.minor)
                + "total memory = {:.3f} GB ; ".format(env.total_memory_in_GB)
                + "name = {:40s}".format(env.name)
            )
        logger.info(first_line)

def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False

@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name