"""Versioned public-input semantics, independent of tensor shapes and weights."""

PPO_INPUT_CONTRACT = "rin.semantic-ppo-input.v2"
LEGACY_PPO_INPUT_CONTRACT = "rin.semantic-ppo-input.v1"
PUBLIC_SHANTEN_CONTRACT = "rin.shanten.pre-draw-or-post-call.v1"
LEGACY_SHANTEN_CONTRACT = "libriichi.shanten.cached.v1"


def native_public_shanten(state, *, contract: str = PUBLIC_SHANTEN_CONTRACT) -> int:
    """Use the own hand before a held draw, or the current post-call hand.

    Completed shapes are reported as tenpai (zero), as in the pretrained
    native feature. Historical exports keep their explicit cached contract.
    """
    if contract == LEGACY_SHANTEN_CONTRACT:
        return int(state.shanten)
    if contract != PUBLIC_SHANTEN_CONTRACT:
        raise ValueError(f"unsupported public shanten contract: {contract}")
    method = getattr(state, "public_shanten", None)
    if not callable(method):
        raise RuntimeError("public_shanten requires the qualified RIN native extension")
    return int(method())
