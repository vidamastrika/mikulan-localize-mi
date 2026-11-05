"""Capacity-only adapter for the pinned SESAME Euclidean neighbour builder.

Keeps neighbour membership, order, connectivity/minimum checks and probabilities.
No installed package is edited. Applied only inside the compatibility runner.
"""
import hashlib
import inspect
import textwrap


def adapt(original):
    source = textwrap.dedent(inspect.getsource(original))
    needle = 'n_max = 100'
    if source.count(needle) != 1:
        raise ValueError('Unrecognized SESAME neighbour implementation; adapter not applied')
    adapted = source.replace(needle, 'n_max = 256', 1)
    namespace = dict(original.__globals__)
    exec(compile(adapted, '<sesame_capacity_256>', 'exec'), namespace)
    return namespace[original.__name__], hashlib.sha256(source.encode()).hexdigest()


def install():
    from sesameeg.utils import smc_utils
    original = smc_utils._compute_euclidean_neigh_matrix
    replacement, digest = adapt(original)
    smc_utils._compute_euclidean_neigh_matrix = replacement
    return dict(original_function_sha256=digest, original_capacity=100,
                compatibility_capacity=256, radius='unchanged automatic 0.01 m',
                minimum_neighbours=3, probability_rule='unchanged')
