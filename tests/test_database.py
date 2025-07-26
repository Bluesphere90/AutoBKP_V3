# tests/test_database.py

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Import database functions
from app.ml.database import (
    create_training_table,
    insert_training_records,
    load_all_training_data,
    get_training_data_count,
    database_exists,
    get_client_database_path
)


@pytest.fixture
def temp_client_id():
    """Fixture tạo temporary client ID cho test."""
    return "test_client_temp"


@pytest.fixture
def sample_records():
    """Sample data để test."""
    return [
        {
            "TaxCode": "MST001",
            "TenHangHoaDichVu": "Máy tính Dell",
            "HachToan": "156",
            "MaHangHoa": "DELL01"
        },
        {
            "TaxCode": "MST002",
            "TenHangHoaDichVu": "Phí vận chuyển",
            "HachToan": "642",
            "MaHangHoa": "VC001"
        },
        {
            "TaxCode": "MST001",
            "TenHangHoaDichVu": "Máy tính Dell",  # Duplicate
            "HachToan": "156",
            "MaHangHoa": "DELL01"
        }
    ]


def test_create_training_table(temp_client_id):
    """Test tạo table."""
    result = create_training_table(temp_client_id)
    assert result == True

    # Kiểm tra file database được tạo
    db_path = get_client_database_path(temp_client_id)
    assert db_path.exists()


def test_insert_training_records(temp_client_id, sample_records):
    """Test insert records và duplicate handling."""
    # Tạo table trước
    create_training_table(temp_client_id)

    # Insert records
    inserted_count = insert_training_records(temp_client_id, sample_records)

    # Should insert 2 records (1 duplicate skipped)
    assert inserted_count == 2

    # Verify total count
    total_count = get_training_data_count(temp_client_id)
    assert total_count == 2


def test_load_all_training_data(temp_client_id, sample_records):
    """Test load data từ database."""
    # Setup data
    create_training_table(temp_client_id)
    insert_training_records(temp_client_id, sample_records)

    # Load data
    df = load_all_training_data(temp_client_id)

    # Verify
    assert df is not None
    assert len(df) == 2  # 2 unique records
    assert "TaxCode" in df.columns
    assert "TenHangHoaDichVu" in df.columns
    assert "HachToan" in df.columns
    assert "MaHangHoa" in df.columns


def test_database_exists(temp_client_id, sample_records):
    """Test kiểm tra database existence."""
    # Initially should not exist
    assert database_exists(temp_client_id) == False

    # After creating and adding data
    create_training_table(temp_client_id)
    insert_training_records(temp_client_id, sample_records)

    assert database_exists(temp_client_id) == True


def test_empty_records(temp_client_id):
    """Test với empty records."""
    create_training_table(temp_client_id)

    # Test empty list
    inserted_count = insert_training_records(temp_client_id, [])
    assert inserted_count == 0

    # Test load from empty database
    df = load_all_training_data(temp_client_id)
    assert df is not None
    assert len(df) == 0


# Cleanup sau mỗi test
@pytest.fixture(autouse=True)
def cleanup_test_database(temp_client_id):
    """Cleanup test database sau mỗi test."""
    yield  # Run test

    # Cleanup
    try:
        db_path = get_client_database_path(temp_client_id)
        if db_path.exists():
            db_path.unlink()

        # Cleanup parent directory if empty
        parent_dir = db_path.parent
        if parent_dir.exists() and not any(parent_dir.iterdir()):
            parent_dir.rmdir()
    except Exception as e:
        print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    # Để chạy trực tiếp file này
    pytest.main([__file__, "-v"])