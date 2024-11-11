import numpy as np;
from scipy.ndimage import zoom;

def unify_and_merge_datasets(file1, file2, label_file1, label_file2):
    """
    Unify two UAV-Human datasets with different shapes into a single dataset, along with their labels.

    Parameters:
    file1 (str): Path to the first .npy file with shape (N1, 3, 300, 17, 2).
    file2 (str): Path to the second .npy file with shape (N2, 3, 64, 17).
    label_file1 (str): Path to the first label .npy file with shape (N1,).
    label_file2 (str): Path to the second label .npy file with shape (N2,).

    Returns:
    tuple: The merged dataset with shape (N1 + N2, 3, 300, 17, 2) and merged labels with shape (N1 + N2,).
    """
    # Load the datasets
    data1 = np.load(file1)  # Shape: (N1, 3, 300, 17, 2)
    data2 = np.load(file2)  # Shape: (N2, 3, 64, 17)
    labels1 = np.load(label_file1)  # Shape: (N1,)
    labels2 = np.load(label_file2)  # Shape: (N2,)

    N2 = data2.shape[0];

    # Expand the data2 to have a person dimension (M=1)
    data2 = np.expand_dims(data2, axis=-1)  # Shape: (N2, 3, 64, 17, 1)

    # Resize the time dimension of data2 from 64 to 300 using interpolation
    data2_resized = np.zeros((N2, 3, 300, 17, 1))
    for i in range(N2):
        for c in range(3):
            for v in range(17):
                data2_resized[i, c, :, v, 0] = zoom(data2[i, c, :, v, 0], zoom=300/64, order=1)[:300]

    # Concatenate along the person dimension to match data1's shape
    data2_final = np.concatenate([data2_resized, np.zeros_like(data2_resized)], axis=-1)  # Shape: (N2, 3, 300, 17, 2)

    # Merge the two datasets
    merged_data = np.concatenate([data1, data2_final], axis=0)  # Shape: (N1 + N2, 3, 300, 17, 2)

    # Merge the labels
    merged_labels = np.concatenate([labels1, labels2], axis=0)  # Shape: (N1 + N2,)

    return merged_data, merged_labels


if __name__ == "__main__":
     merged_data, merged_labels = unify_and_merge_datasets("../data_c/train_joint.npy", "../data_c/gan/gen_data.npy", "../data_c/train_label.npy", "../data_c/gan/gen_labels.npy");
     with open("new_train.npy", "wb") as f:
        np.save(f, merged_data);
     with open("new_label.npy", "wb") as f:
        np.save(f, merged_labels);
