from .expert import UpstreamExpert as _UpstreamExpert


def null_upstream(*args, **kwargs):
    """

    """
    return _UpstreamExpert(*args, **kwargs)
