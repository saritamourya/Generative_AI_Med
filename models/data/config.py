import torch
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    LoadImaged,
    Invertd,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

params = {
    'WINDOW_WIDTH': 70,
    'WINDOW_LEVEL': 30,
    'num_pool': 100, #number of images generated in image pool
    'roi_size': [32, 32, 32], #determines the patch size
    'pixdim':(0.86, 0.86, 2.50), #resampling pixel dimensions
    'imgA_intensity_range': (-1024, 3071), #range of intensities for nomalization to range [0,1]
    'imgB_intensity_range': (0, 20)
}

# Transformations for dynamic loading and sampling of Nifti files
train_transforms = Compose([
    # Carica le immagini mediche da file nifti in tensori pytorch
    LoadImaged(keys=['A', 'B']),
    # Assicura che i canali delle immagini siano posizionati come primo asse nei tensori
    EnsureChannelFirstd(keys=['A', 'B']),
    # Orientationd(keys=['imgA', 'imgB'], axcodes='RAS'),
    # Esegue il ritaglio dell'immagine per rimuovere le regioni "vuote" o non rilevanti dell'immagine
    #CropForegroundd(keys=['A'], source_key='A'),
    #CropForegroundd(keys=['B'], source_key='B'),

    # Imposta le dimensioni dei pixel delle immagini secondo i valori specificati in params ['pixdim']
    #Spacingd(keys=['A', 'B'], pixdim=params['pixdim'], mode=("bilinear", "bilinear")),

    # Normalizza l'intensità dei pixel delle immagini in un determinato intervallo
    ScaleIntensityRanged(keys=['A'], a_min=params['imgA_intensity_range'][0], a_max=params['imgA_intensity_range'][1], b_min=0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=['B'], a_min=params['imgB_intensity_range'][0], a_max=params['imgB_intensity_range'][1], b_min=0, b_max=1.0, clip=True),
    # Esegue il ritaglio casuale dell'immagine in base alle dimensioni specificate in params['roi_size']
    RandSpatialCropd(keys=['A', 'B'], roi_size=params['roi_size'], random_size=False, random_center=True),
    #RandSpatialCropd(keys=['imgB'], roi_size=params['roi_size'], random_size=False, random_center=True),

    # Esegue il padding delle immagini per farle coincidere con le dimensioni specificate in params['roi_size']
    SpatialPadd(keys=["A", "B"], spatial_size=params['roi_size']),
])

test_transforms = Compose([
    LoadImaged(keys=['A', 'B']),
    EnsureChannelFirstd(keys=['A', 'B']),
    #     Orientationd(keys=['imgA', 'imgB'], axcodes='RAS'),
    #CropForegroundd(keys=['A'], source_key='A'),
    #CropForegroundd(keys=['B'], source_key='B'),
    #Spacingd(keys=['A', 'B'], pixdim=params['pixdim'], mode=("bilinear", "bilinear")),
    ScaleIntensityRanged(keys=['A'], a_min=params['imgA_intensity_range'][0], a_max=params['imgA_intensity_range'][1], b_min=0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=['B'], a_min=params['imgB_intensity_range'][0], a_max=params['imgB_intensity_range'][1], b_min=0, b_max=1.0, clip=True),
    #RandSpatialCropd(keys=['A'], roi_size=params['roi_size'], random_size=False, random_center=True),
    #RandSpatialCropd(keys=['B'], roi_size=params['roi_size'], random_size=False, random_center=True),
    SpatialPadd(keys=["A", "B"], spatial_size=params['roi_size']),
])
