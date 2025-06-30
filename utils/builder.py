from .registry import build, Registry

METRICS = Registry("Metrics")
VISUAL = Registry("Visual")
# MLLM = Registry("Mllm")
IMAGEGEN = Registry("Imagegen")


def build_metrics(cfg, *args, **kwargs):
    return build(cfg, METRICS, *args, **kwargs)


def build_visual(cfg, *args, **kwargs):
    return build(cfg, VISUAL, *args, **kwargs)


# def build_mllm(cfg, *args, **kwargs):
#     return build(cfg, MLLM, *args, **kwargs)


def build_imagegen(cfg, *args, **kwargs):
    return build(cfg, IMAGEGEN, *args, **kwargs)
