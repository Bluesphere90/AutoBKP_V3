# app/ml/database.py - File mới

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import app.core.config as config
from app.ml.utils import get_client_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_client_database_path(client_id: str) -> Path:
    """Lấy đường dẫn đến SQLite database của client."""
    client_path = get_client_data_path(client_id)
    return client_path / "training_data.db"


@contextmanager
def get_database_connection(client_id: str):
    """Context manager để quản lý SQLite connection."""
    db_path = get_client_database_path(client_id)

    # Tạo thư mục nếu chưa có
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error for client {client_id}: {e}")
        raise
    finally:
        if conn:
            conn.close()


def create_training_table(client_id: str) -> bool:
    """Tạo bảng training_records nếu chưa tồn tại."""
    try:
        with get_database_connection(client_id) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tax_code TEXT NOT NULL,
                    ten_hang_hoa TEXT NOT NULL,
                    hach_toan TEXT,
                    ma_hang_hoa TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(tax_code, ten_hang_hoa)
                )
            """)

            # Tạo index để query nhanh
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_input 
                ON training_records(tax_code, ten_hang_hoa)
            """)

            conn.commit()
            logger.info(f"Training table created/verified for client {client_id}")
            return True

    except Exception as e:
        logger.error(f"Failed to create training table for client {client_id}: {e}")
        return False


def insert_training_records(client_id: str, records: List[Dict[str, Any]]) -> int:
    """
    Insert training records vào database, skip duplicates.

    Returns:
        Số lượng records được insert thành công
    """
    if not records:
        return 0

    # Ensure table exists
    if not create_training_table(client_id):
        return 0

    inserted_count = 0

    try:
        with get_database_connection(client_id) as conn:
            for record in records:
                try:
                    # Check if record already exists
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM training_records 
                        WHERE tax_code = ? AND ten_hang_hoa = ?
                    """, (
                        record.get('TaxCode', ''),
                        record.get('TenHangHoaDichVu', '')
                    ))

                    exists = cursor.fetchone()[0] > 0

                    if not exists:
                        # Insert new record
                        conn.execute("""
                            INSERT INTO training_records 
                            (tax_code, ten_hang_hoa, hach_toan, ma_hang_hoa)
                            VALUES (?, ?, ?, ?)
                        """, (
                            record.get('TaxCode', ''),
                            record.get('TenHangHoaDichVu', ''),
                            record.get('HachToan', ''),
                            record.get('MaHangHoa', '')
                        ))
                        inserted_count += 1

                except sqlite3.Error as e:
                    logger.warning(f"Failed to insert record {record}: {e}")
                    continue

            conn.commit()
            logger.info(f"Inserted {inserted_count}/{len(records)} new records for client {client_id}")

    except Exception as e:
        logger.error(f"Failed to insert training records for client {client_id}: {e}")
        return 0

    return inserted_count


def load_all_training_data(client_id: str) -> Optional[pd.DataFrame]:
    """Load toàn bộ training data từ database."""
    try:
        with get_database_connection(client_id) as conn:
            # First check if table exists and has data
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='training_records'
            """)

            if not cursor.fetchone():
                logger.warning(f"Table training_records does not exist for client {client_id}")
                return pd.DataFrame()

            # Check record count
            cursor = conn.execute("SELECT COUNT(*) FROM training_records")
            count = cursor.fetchone()[0]
            logger.info(f"Found {count} records in database for client {client_id}")

            if count == 0:
                return pd.DataFrame()

            # Map column names để consistent với code hiện tại
            df = pd.read_sql_query("""
                SELECT 
                    tax_code as TaxCode,
                    ten_hang_hoa as TenHangHoaDichVu,
                    hach_toan as HachToan,
                    ma_hang_hoa as MaHangHoa,
                    created_at
                FROM training_records
                ORDER BY created_at ASC
            """, conn)

            logger.info(f"Loaded {len(df)} training records for client {client_id}")
            return df

    except Exception as e:
        logger.error(f"Failed to load training data for client {client_id}: {e}")
        return None


def get_training_data_count(client_id: str) -> int:
    """Đếm số lượng records trong database."""
    try:
        with get_database_connection(client_id) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM training_records")
            count = cursor.fetchone()[0]
            return count
    except Exception as e:
        logger.error(f"Failed to count training data for client {client_id}: {e}")
        return 0


def database_exists(client_id: str) -> bool:
    """Kiểm tra xem database có tồn tại và có data không."""
    db_path = get_client_database_path(client_id)
    if not db_path.exists():
        return False

    try:
        count = get_training_data_count(client_id)
        return count > 0
    except:
        return False