import torch

from gluemap.ff_inference.local_inference import LocalInference


class VGGTOmegaLocalInference(LocalInference):
    """Local inference for VGGT-Omega backbone."""

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        """Run VGGT-Omega on a batch of images.

        Args:
            batch: Dict with ``"images"`` of shape ``(B, N, 3, H, W)``.

        Returns:
            The model prediction dict augmented with ``extrinsics`` and
            ``intrinsics`` decoded from VGGT-Omega's pose encoding.
        """
        images = batch["images"].to(self.device).contiguous()

        predictions = self.model(images)

        from vggt_omega.utils.pose_enc import encoding_to_camera

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            image_size_hw=images.shape[-2:],
        )
        predictions["extrinsics"] = extrinsics
        predictions["intrinsics"] = intrinsics

        return predictions
