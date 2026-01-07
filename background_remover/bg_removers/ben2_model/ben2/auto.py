from background_remover.bg_removers.ben2_model.ben2.modeling_ben2 import BEN_Base
from huggingface_hub import model_info, PyTorchModelHubMixin

from functools import wraps


def set_doc():
    """
    Decorator to set the docstring of the from_pretrained method to the one of the AutoModel class.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = PyTorchModelHubMixin.from_pretrained.__doc__
        return wrapper

    return decorator


class AutoModel:
    @classmethod
    @set_doc()
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        tags = model_info(pretrained_model_name_or_path).tags
        if "BEN2" in tags:  # using tags to determine which model architectire to call
            return BEN_Base.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
