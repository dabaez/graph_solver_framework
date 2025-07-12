import framework.dataset_creators  # noqa: F401
import framework.feature_extractors  # noqa: F401
import framework.solvers  # noqa: F401

from .utils import (
    list_registered_dataset_creators,
    list_registered_dataset_feature_extractors,
    list_registered_dataset_solvers,
)

if __name__ == "__main__":
    print("Registered Dataset Creators:")
    print(list_registered_dataset_creators())

    print("\nRegistered Dataset Solvers:")
    print(list_registered_dataset_solvers())

    print("\nRegistered Dataset Feature Extractors:")
    print(list_registered_dataset_feature_extractors())
