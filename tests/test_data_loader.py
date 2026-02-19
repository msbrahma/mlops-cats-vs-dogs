
import os
from src.data_loader import get_data_loaders

def test_data_loader_creation():
    data_dir = "data/processed"

    assert os.path.exists(data_dir), "Processed data directory does not exist"

    train_loader, val_loader, test_loader = get_data_loaders(data_dir)

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
