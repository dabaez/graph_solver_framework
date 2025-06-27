from typing import Protocol, runtime_checkable

from framework.core.graph import Dataset, Feature


@runtime_checkable
class DatasetFeatureExtractor(Protocol):
    """Base class for dataset feature extractors."""

    def extract_features(self, dataset: Dataset) -> list[list[Feature]]:
        """
        Extract features from the dataset.

        :param dataset: The dataset from which to extract features.
        :return: A list of lists of features extracted from the dataset.
        """
        ...
