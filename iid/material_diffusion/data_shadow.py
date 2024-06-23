import os

import torch
from batch import Batch

from iid.data import IIDDataset
from iid.utils import TrainStage


class ShadowDataset(IIDDataset):
    FEATURES = ["origin", "shadow_free"]
    DERIVED_FEATURES = []

    def load_dataset(self, allow_missing_features=False):
        # Collect the data
        data = Batch()

        # Collect the features
        self.module_logger.debug("Collecting features")
        data['samples'] = Batch(default=Batch)
        data['sample_ids'] = []

        root_path = "/home/disk1/Dataset/complete_dataset/render_data_v2/train/"
        if self.stage == TrainStage.Training:
            root_path = "/home/disk1/Dataset/complete_dataset/render_data_v2/train/"
        elif self.stage == TrainStage.Validation:
            root_path = "/home/disk2/dataset/render_data/test/"
        elif self.stage == TrainStage.Test:
            root_path = "/home/disk2/dataset/render_data/test/"
        else:
            raise ValueError(f"Invalid stage {self.stage}!")

        for feature in self.FEATURES:
            feature_folder_path = os.path.join(root_path, feature)
            assert os.path.exists(feature_folder_path), f"Missing {feature} data."
            for file_name in sorted(os.listdir(feature_folder_path)):
                if "_" not in file_name:
                    continue

                sample_id = file_name.split('.')[0].split('_')[1]

                if sample_id not in data['samples']:
                    data['sample_ids'].append(sample_id)
                    for feature in self.features_to_include:
                        data['samples'][sample_id][feature] = os.path.join(feature, file_name)

        # Sanity check
        lengths = [len(list(data['samples'][sample_id].keys())) for sample_id in data['samples'].keys()]
        assert all([lengths[0] == l for l in lengths]), "Missing feature!"

        return data
