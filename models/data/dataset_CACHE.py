import csv
import os

from monai.data import Dataset, DataLoader, CacheDataset
import data.config


class MioDataset(Dataset):
    def __init__(self, opt, folder_paths):
        self.opt = opt
        self.folder_paths = folder_paths
        self.files_A, self.files_B = self._load_files()

    def _append_pair_if_exists(
        self,
        files_A,
        files_B,
        ct_path,
        pet_path,
        folder_path,
    ):
        if os.path.exists(ct_path) and os.path.exists(pet_path):
            files_A.append(ct_path)
            files_B.append(pet_path)
        else:
            print(f"[WARNING] Missing file(s) in folder: {folder_path}")
            print(f"  CT:  {ct_path}")
            print(f"  PET: {pet_path}")

    def _load_files(self):
        files_A = []
        files_B = []

        for folder_path in self.folder_paths:
            ct_path = os.path.join(folder_path, "CT_cropped.nii.gz")
            pet_path = os.path.join(folder_path, "PET_cropped.nii.gz")

            self._append_pair_if_exists(
                files_A,
                files_B,
                ct_path,
                pet_path,
                folder_path,
            )

        return files_A, files_B

    def __getitem__(self, index):
        img_path_A = self.files_A[index]
        img_path_B = self.files_B[index]

        # Example: PETCT_0af7ffe12a/96698/CT_cropped.nii.gz
        scan_folder = os.path.dirname(img_path_A)
        patient_id = os.path.basename(os.path.dirname(scan_folder))
        scan_id = os.path.basename(scan_folder)

        a_name = f"{patient_id}_{scan_id}_CT"
        b_name = f"{patient_id}_{scan_id}_PET"

        return {
            "A": img_path_A,
            "B": img_path_B,
            "A_paths": img_path_A,
            "B_paths": img_path_B,
            "a_name": a_name,
            "b_name": b_name,
        }

    def __len__(self):
        return len(self.files_B)


def CreateDataloader(
    opt,
    shuffle=True,
    num_workers=0,
    drop_last=False,
    cache=False,
):
    folder_paths = []

    csv_path = (
        f"{opt.dataroot}/{opt.phase}_{opt.district}.csv"
        if opt.phase == "train"
        else f"{opt.dataroot}/{opt.phase}.csv"
    )

    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row and row[0].strip():
                folder_paths.append(os.path.normpath(row[0].strip()))

    if not folder_paths:
        print(f"[INFO] No folder paths found in file {csv_path}")
        return None

    mio_dataset = MioDataset(opt, folder_paths)

    if len(mio_dataset) == 0:
        raise ValueError(
            f"No complete CT/PET pairs found using {csv_path}"
        )

    if cache:
        ds = CacheDataset(
            data=mio_dataset,
            transform=(
                data.config.train_transforms
                if opt.phase == "train"
                else data.config.test_transforms
            ),
        )
    else:
        ds = Dataset(
            data=mio_dataset,
            transform=(
                data.config.train_transforms
                if opt.phase == "train"
                else data.config.test_transforms
            ),
        )

    data_loader = DataLoader(
        ds,
        batch_size=opt.batchSize,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=True,
    )

    return data_loader