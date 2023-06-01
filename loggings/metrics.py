import contextlib
import uuid
from collections import defaultdict, OrderedDict
from typing import Callable, List, Optional

from .meters import *

_aggregators = OrderedDict()
_active_aggregators = OrderedDict()
_active_aggregators_cnt = defaultdict(lambda: 0)

def reset() -> None:
    _aggregators.clear()
    _active_aggregators.clear()
    _active_aggregators_cnt.clear()

    _aggregators['default'] = MetersDict()
    _active_aggregators['default'] = _aggregators['default']
    _active_aggregators_cnt['default'] = 1

reset()

@contextlib.contextmanager
def aggregate(name=None, new_root=False):
    if name is None:
        name = str(uuid.uuid4())
        assert name not in _aggregators
        agg = MetersDict()
    else:
        assert name != 'default'
        agg = _aggregators.setdefault(name, MetersDict())
    
    if new_root:
        backup_aggregators = _active_aggregators.copy()
        _active_aggregators.clear()
        backup_aggregators_cnt = _active_aggregators_cnt.copy()
        _active_aggregators_cnt.clear()
    
    _active_aggregators[name] = agg
    _active_aggregators_cnt[name] += 1

    yield agg

    _active_aggregators_cnt[name] -= 1
    if _active_aggregators_cnt[name] == 0 and name in _active_aggregators:
        del _active_aggregators[name]
    
    if new_root:
        _active_aggregators.clear()
        _active_aggregators.update(backup_aggregators)
        _active_aggregators_cnt.clear()
        _active_aggregators_cnt.update(backup_aggregators_cnt)

def get_active_aggregators() -> List[MetersDict]:
    return list(_active_aggregators.values())

def log_scalar(
    key,
    value,
    weight=1,
    priority=10,
    round=None
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, AverageMeter(round=round), priority)
        agg[key].update(value, weight)

def log_scalar_sum(
    key,
    value,
    priority=10,
    round=None
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, SumMeter(round=round), priority)
        agg[key].update(value)

def log_derived(
    key,
    fn: Callable[[MetersDict], float],
    priority=20
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, MetersDict._DerivedMeter(fn), priority)

def log_speed(
    key,
    value,
    priority=30,
    round=None
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, TimeMeter(round=round), priority)
            agg[key].reset()
        else:
            agg[key].update(value)

def log_start_time(
    key,
    priority=40,
    round=None
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, StopwatchMeter(round=round), priority)
    agg[key].start()

def log_stop_time(
    key,
    weight=0.0,
    prehook=None
):
    for agg in get_active_aggregators():
        if key in agg:
            agg[key].stop(weight, prehook)

def log_custom(
    new_meter_fn: Callable[[], Meter],
    key,
    *args,
    priority=50,
    **kwargs,
):
    for agg in get_active_aggregators():
        if key not in agg:
            agg.add_meter(key, new_meter_fn(), priority)
        agg[key].update(*args, **kwargs)

def reset_meter(name, key) -> None:
    meter = get_meter(name, key)
    if meter is not None:
        meter.reset()

def reset_meters(name) -> None:
    meters = get_meters(name)
    if meters is not None:
        meters.reset()

def get_meter(name, key) -> Meter:
    if name not in _aggregators:
        return None
    return _aggregators[name].get(key, None)

def get_meters(name) -> MetersDict:
    return _aggregators.get(name, None)

def get_smoothed_values(name) -> Dict[str, float]:
    return _aggregators[name].get_smoothed_values()

def state_dict():
    return OrderedDict([(name, agg.state_dict()) for name, agg in _aggregators.items()])