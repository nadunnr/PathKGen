from pathlib import Path
from dataset import SlideEmbeddingDataset

if __name__ == "__main__":
    root_dir = "/home/nadun/wd/TRIDENT/trident_processed/40x_512px_0px_overlap/slide_features_titan"

    dataset = SlideEmbeddingDataset(root_dir)

    # collect all class labels
    all_classes = []
    for _, cls in dataset:
        all_classes.append(cls)

    # print results
    print("Total slides:", len(all_classes))
    print("All classes:")
    for c in all_classes:
        print(c)

    # unique classes
    unique_classes = sorted(set(all_classes))
    print("\nUnique classes:", unique_classes)
    print("Number of unique classes:", len(unique_classes))