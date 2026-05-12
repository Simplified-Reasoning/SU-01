from slime.utils.types import Sample

__all__ = ["mask_overlong_samples"]

def mask_overlong_samples(args, data: list[list[Sample]], **kwargs):
    for group in data:
        samples = group if not isinstance(group[0], list) else [s for subgroup in group for s in subgroup]
        for sample in samples:
            if sample.status == Sample.Status.TRUNCATED:
                sample.remove_sample = True