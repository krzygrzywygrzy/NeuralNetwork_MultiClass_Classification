import numpy
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import MultiClassClassificationModel

# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1. Generate multi-class data
X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, random_state=RANDOM_SEED,
                            cluster_std=1.5)

# 2. Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# 3. Split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED
                                                                        )

# 4. Visualise, visualise, visualise
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.title('Dataset')

# 5. Create and train the model
model_0 = MultiClassClassificationModel(input_features=NUM_FEATURES, output_features=NUM_CLASSES)
# default setup with 1000 epochs and learning_rate = 0.01, gives 99.5% accuracy on test set
model_0.train_model(X_blob_train, X_blob_test, y_blob_train, y_blob_test)

# 6. Predict and show some statistics
with torch.no_grad():
    y_logits = model_0(X_blob_train)
    predictions = torch.softmax(y_logits, dim=1).argmax(dim=1)

    labels, counts = numpy.unique(predictions.numpy(), return_counts=True)

    # Visualise, visualise, visualise
    plt.figure()
    plt.bar(labels, counts)
    plt.xlabel('Class')
    plt.ylabel('Amount')
    plt.show()
