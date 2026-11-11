import torch
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRangePercentilesd,
)

params = {
    'num_pool': 100,
    'roi_size': [128, 128, 128],  # Adjust for GPU memory and latent dimensions
    'pixdim': (1.0, 1.0, 1.0),   # Informational; no resampling below
    'lower_percentile': 0.5,
    'upper_percentile': 99.5,
}

# A: T1w source image
# B: T2f / T2-FLAIR target image
#
# Expected dataset entry:
# {'A': '/path/to/t1w.nii.gz', 'B': '/path/to/t2f.nii.gz'}

train_transforms = Compose([
    LoadImaged(keys=['A', 'B']),

    EnsureChannelFirstd(keys=['A', 'B']),

    Orientationd(keys=['A', 'B'], axcodes='RAS'),

    # Normalize each modality independently to [0, 1].
    # Percentiles here include zero-valued background voxels.
    ScaleIntensityRangePercentilesd(
        keys=['A', 'B'],
        lower=params['lower_percentile'],
        upper=params['upper_percentile'],
        b_min=0.0,
        b_max=1.0,
        clip=True,
        channel_wise=True,
    ),

    # Pad before cropping to guarantee the requested patch size.
    SpatialPadd(
        keys=['A', 'B'],
        spatial_size=params['roi_size'],
        mode='constant',
    ),

    # Apply the same crop coordinates to both registered modalities.
    RandSpatialCropd(
        keys=['A', 'B'],
        roi_size=params['roi_size'],
        random_size=False,
        random_center=True,
    ),

    EnsureTyped(keys=['A', 'B'], dtype=torch.float32),
])

test_transforms = Compose([
    LoadImaged(keys=['A', 'B']),

    EnsureChannelFirstd(keys=['A', 'B']),

    Orientationd(keys=['A', 'B'], axcodes='RAS'),

    ScaleIntensityRangePercentilesd(
        keys=['A', 'B'],
        lower=params['lower_percentile'],
        upper=params['upper_percentile'],
        b_min=0.0,
        b_max=1.0,
        clip=True,
        channel_wise=True,
    ),

    SpatialPadd(
        keys=['A', 'B'],
        spatial_size=params['roi_size'],
        mode='constant',
    ),

    EnsureTyped(keys=['A', 'B'], dtype=torch.float32),
])