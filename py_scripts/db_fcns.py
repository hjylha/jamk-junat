import sqlite3
from pathlib import Path

import pandas as pd


THIS_FILE_PATH = Path(__file__)
# "data/db.db"
DB_PATH = THIS_FILE_PATH.parent.parent / "data" / "db.db"
if not DB_PATH.parent.exists():
    DB_PATH.parent.mkdir()

try:
    from .extra_data_paths import get_extra_train_data_path
    EXTRA_DB_PATH = get_extra_train_data_path() / "juna.db"
    if not EXTRA_DB_PATH.parent.exists():
        EXTRA_DB_PATH.parent.mkdir()
except ModuleNotFoundError:
    print(f"No path for extra data found, using regular one instead ({DB_PATH}).")
    EXTRA_DB_PATH = DB_PATH


def save_df_to_db(df, table_name, to_extra=False, if_exists_action="fail", db_path=None):
    if db_path is None:
        db_path = EXTRA_DB_PATH if to_extra else DB_PATH
    with sqlite3.connect(db_path) as conn:
        try:
            df.to_sql(name=table_name, con=conn, if_exists=if_exists_action, index=False)
            conn.commit()
        except ValueError:
            print(f"Table {table_name} already exists")



def get_df_from_db(table_name, from_extra=False, db_path=None):
    if db_path is None:
        db_path = EXTRA_DB_PATH if from_extra else DB_PATH
    try:
        with sqlite3.connect(db_path) as conn:
            try:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            except pd.io.sql.DatabaseError:
                print(f"probably table {table_name} does not exist")
                return
    except sqlite3.OperationalError:
        print(f"probably no database at {db_path}")
        return

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df



if __name__ == "__main__":
    print(f"{THIS_FILE_PATH=}")
    print(f"{DB_PATH=}")
    print(f"{EXTRA_DB_PATH=}")


