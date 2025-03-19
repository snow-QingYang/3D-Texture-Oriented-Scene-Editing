# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model for GaussCtrl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, Union, Dict, List, Optional

import torch, math
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    interlevel_loss,
)
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, projection_matrix
from gsplat.sh import num_sh_bases, spherical_harmonics
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components import renderers
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from nerfstudio.data.scene_box import OrientedBox
from gstex.gstex import GStexModel, GStexModelConfig
@dataclass
class GaussCtrlModelConfig(SplatfactoModelConfig):
    """Configuration for the GaussCtrl."""
    _target: Type = field(default_factory=lambda: GaussCtrlModel)
    use_lpips: bool = True
    """Whether to use LPIPS loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 32
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = 1.0
    """Multiplier for LPIPS loss."""

class GaussCtrlModel(SplatfactoModel):
    """Model for GaussCtrl."""

    config: GaussCtrlModelConfig

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        self.training = False
        outs = self.get_outputs(camera.to(self.device))
        # outputs = {}
        # outputs["rgb"] = outs["rgb"]
        # outputs["depth"] = outs["depth"]
        # outputs["accumulation"] = outs["accumulation"]
        
        self.training = True
        return outs  # type: ignore

