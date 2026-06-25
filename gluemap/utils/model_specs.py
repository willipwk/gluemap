"""Backbone-specific model metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneInputSpec:
    """Image preprocessing settings expected by a feed-forward backbone."""

    image_size: int
    patch_size: int


BACKBONE_INPUT_SPECS = {
    "pi3": BackboneInputSpec(image_size=518, patch_size=14),
    "pi3x": BackboneInputSpec(image_size=518, patch_size=14),
    "vggt": BackboneInputSpec(image_size=518, patch_size=14),
    "map_anything": BackboneInputSpec(image_size=518, patch_size=14),
    "vggt_omega": BackboneInputSpec(image_size=512, patch_size=16),
}


def get_backbone_input_spec(model_type: str) -> BackboneInputSpec:
    """Return preprocessing settings for ``model_type``."""
    return BACKBONE_INPUT_SPECS.get(
        model_type,
        BackboneInputSpec(image_size=518, patch_size=14),
    )
