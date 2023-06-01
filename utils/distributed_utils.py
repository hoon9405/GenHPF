import logging
import os
import random
import socket
import struct
import pickle
import warnings

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

def is_master(args):
    return args.distributed_rank == 0

def get_rank(group):
    return dist.get_rank(group=group)

def get_world_size(group):
    if torch.distributed.is_initialized():
        return dist.get_world_size(group=group)
    else:
        return 1

def get_global_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None

def get_global_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def get_global_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    return get_global_group()

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_rank(get_data_parallel_group())

def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_world_size(get_data_parallel_group())

def infer_init_method(args):
    assert (
        args.world_size <= torch.cuda.device_count()
    ), f"world size is {args.world_size} but have {torch.cuda.device_count()} available devices"
    port = random.randint(10000, 20000)
    args.distributed_init_method = "tcp://localhost:{port}".format(port=port)

def distributed_init(args):
    if dist.is_available() and dist.is_initialized():
        warnings.warn(
            'Distributed is already initialized'
        )
    else:
        logger.info(
            'distributed init (rank {}): {}'.format(
                args.distributed_rank,
                args.distributed_init_method
            )
        )
        dist.init_process_group(
            backend='nccl',
            init_method=args.distributed_init_method,
            world_size=args.world_size,
            rank=args.distributed_rank
        )
        logger.info(
            'initialized host {} as rank {}'.format(
                socket.gethostname(),
                args.distributed_rank
            )
        )

        #perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    args.distributed_rank = dist.get_rank()

    if is_master(args):
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    return args.distributed_rank

def all_reduce(tensor, group, op = "sum"):
    if op == "sum":
        op = dist.ReduceOp.SUM
    elif op == "max":
        op = dist.ReduceOp.MAX
    else:
        raise NotImplementedError
    
    dist.all_reduce(tensor, op = op, group = group)
    
    return tensor

def all_gather_list(data, group = None, max_size = 32768):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable and any CUDA tensors will be moved
    to CPU and returned on CPU as well.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    import utils.utils as utils

    if group is None:
        group = get_global_group()
    rank = get_rank(group = group)
    world_size = get_world_size(group = group)

    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, "_buffer")
        or all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    data = utils.move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4 # size of header that contains the length of the encoded data
    size = header_size + enc_size
    if size > max_size:
        raise ValueError(
            "encoded data size ({}) exceeds max_size ({})".format(size, max_size)
        )
    
    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start : start + size].copy_(cpu_buffer[:size])
    
    all_reduce(buffer, group = group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            (enc_size,) = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(
                    pickle.loads(
                        bytes(out_buffer[header_size : header_size + enc_size].tolist())
                    )
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data. "
            # "Try rerunning with --ddp-backend=legacy_ddp and see if that helps."
        )

def distributed_main(i, main, args, kwargs):
    args.device_id = i
    if torch.cuda.is_available():
        torch.cuda.set_device(i)
    if args.distributed_rank is None:
        args.distributed_rank = i
    
    args.distributed_rank = distributed_init(args)

    main(args, **kwargs)

    if dist.is_initialized():
        dist.barrier(get_global_group())

def call_main(args, main, **kwargs):
    if args.world_size > 1:
        infer_init_method(args)
        args.distributed_rank = None
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(main, args, kwargs),
            nprocs=min(
                torch.cuda.device_count(),
                args.world_size
            ),
            join=True
        )
    else:
        args.distributed_rank = 0
        main(args, **kwargs)

def batch_all_gather(tensor, group, return_tensor=False):
    """Perform an all-gather operation considering tensors with different batch size"""
    world_size = get_world_size(group=group)
    rank = get_rank(group=group)

    size_list = [
        tensor.new_zeros(tensor.dim(), dtype=torch.int64) for _ in range(world_size)
    ]
    local_size = tensor.new_tensor(tensor.shape, dtype=torch.int64)
    dist.all_gather(size_list, local_size, group=group)

    max_size = torch.stack(size_list).max(dim=0)[0][0]
    size_offsets = [max_size - size[0] for size in size_list]

    if local_size[0] != max_size:
        offset = torch.cat(
            (
                tensor.new_tensor([max_size - local_size[0]]),
                local_size[1:]
            )
        )
        padding = tensor.new_zeros(tuple(int(dim) for dim in offset), dtype=torch.uint8)
        tensor = torch.cat((tensor, padding), dim=0)

    tensor_list = [
        tensor if i == rank else torch.empty_like(tensor) for i in range(world_size)
    ]
    dist.all_gather(tensor_list, tensor, group=group)
    tensor_list = [
        tensor[:max_size-size_offsets[i]] for i, tensor in enumerate(tensor_list)
    ]
    if return_tensor:
        return torch.stack(tensor_list, dim=0)
    else:
        return tensor_list


def all_gather(tensor, group, return_tensor = False):
    """Perform an all-gather operation."""
    world_size = get_world_size(group = group)
    rank = get_rank(group = group)
    tensor_list = [
        tensor if i == rank else torch.empty_like(tensor) for i in range(world_size)
    ]
    dist.all_gather(tensor_list, tensor, group = group)
    if return_tensor:
        return torch.stack(tensor_list, dim = 0)
    else:
        return tensor_list
