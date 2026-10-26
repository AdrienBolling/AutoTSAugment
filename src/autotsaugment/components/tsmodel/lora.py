"""LoRA adapters on a Timestral network, injected in place with peft.

:class:`timestral.TimestralModel` has no ``forward``, so a ``PeftModel`` wrapper adds nothing:
its ``forward`` cannot run the network. :func:`peft.inject_adapter_in_model` replaces the
target linears with LoRA layers inside the network itself and wraps the modules to save, so
the code that calls the submodules of the network runs the adapters without change.
"""

import math

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from torch import nn


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    targets: tuple[str, ...],
    fully_trainable: tuple[str, ...],
) -> None:
    """Inject LoRA adapters into ``model`` in place and freeze the other parameters.

    After the call, only the adapters and the copies of the ``fully_trainable`` modules
    require gradients.

    Args:
        model: The network. It keeps its type and its methods.
        rank: LoRA rank r.
        alpha: LoRA scale numerator; the update is scaled by ``alpha / rank``.
        dropout: Dropout on the adapter input.
        targets: Names of the linear modules that get an adapter, for example ``wQKV``.
        fully_trainable: Names of the modules trained in full, for example ``head``.
    """
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(targets),
        modules_to_save=list(fully_trainable) or None,
    )
    inject_adapter_in_model(config, model)


def reset_adapters(model: nn.Module, seed: int) -> None:
    """Re-initialize every LoRA adapter of ``model`` from ``seed``, as peft does.

    ``lora_A`` gets the default init of ``nn.Linear`` (Kaiming uniform), drawn on the CPU
    from a generator seeded with ``seed``, so the values do not depend on the device.
    ``lora_B`` gets zeros, so the adapted network equals the base network.

    Args:
        model: A network with adapters from :func:`inject_lora`.
        seed: Seed of the ``lora_A`` values.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoraLayer):
                continue
            for adapter in sorted(module.lora_A.keys()):
                for param in module.lora_A[adapter].parameters():
                    values = torch.empty(param.shape, dtype=torch.float32)
                    nn.init.kaiming_uniform_(values, a=math.sqrt(5), generator=generator)
                    param.copy_(values)
                for param in module.lora_B[adapter].parameters():
                    param.zero_()
