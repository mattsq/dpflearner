from outdist.data.datasets import make_dataset

def test_make_dataset_splits_sum():
    train_ds, val_ds, test_ds = make_dataset("dummy", n_samples=50, splits=(0.6, 0.2, 0.2))
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == 50


def test_synthetic_dataset_shapes():
    train_ds, _, _ = make_dataset("synthetic", n_samples=30, n_features=4)
    x, y = train_ds[0]
    assert x.shape == (4,)
    assert y.dim() == 0
