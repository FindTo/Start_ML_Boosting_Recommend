import pandas as pd
from learn_model import get_user_df
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, REAL, VARCHAR
from io import StringIO
import os
import gc
from tqdm import tqdm
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Type map for PostgreSQL table
TYPE_MAP = {
            'user_id': Integer,                 # int32
            'post_id': Integer,                 # int32
            'target': Integer,                  # int32
            'topic': VARCHAR(50),               # string
            'cluster_1': Integer,               # int32
            'cluster_2': Integer,
            'cluster_3': Integer,
            'cluster_4': Integer,
            'cluster_5': Integer,
            'cluster_6': Integer,
            'cluster_7': Integer,
            'cluster_8': Integer,
            'cluster_9': Integer,
            'text_length': Integer,             # int32
            'gender': Integer,                  # int32
            'age': Integer,                     # int32
            'country': VARCHAR(50),             # string
            'exp_group': Integer,               # int32
            'city_capital': Integer,            # int32
            'post_likes': Integer,              # int32
            'post_views': Integer,              # int32
            'hour': Integer,                    # int32
            'month': Integer,                   # int32
            'day': Integer,                     # int32
            'time_indicator': Integer,          # int32
            'main_topic_liked': VARCHAR(50),    # string
            'main_topic_viewed': VARCHAR(50),   # string
            'views_per_user': Integer,          # int32
            'likes_per_user': Integer           # int32
        }


# Get full DF with all users, relying on the DF for model learning
def get_user_features() -> pd.DataFrame:

    # Download original 'user' dataframe
    user = get_user_df()
    print(user.user_id.nunique())

    # Download master DF for learning
    data = load_features(TYPE_MAP)

    print(data.shape)
    print(data.head())
    print(data.columns)

    # Create 'city' boolean feature as for the learning DF
    capitals = ['Moscow',
                'Saint Petersburg',
                'Kyiv',
                'Minsk',
                'Baku',
                'Almaty',
                'Astana',
                'Helsinki',
                'Istanbul',
                'Ankara',
                'Riga',
                'Nicosia',
                'Limassol',
                'Zurich',
                'Bern',
                'Tallin']
    user['city'] = user.city.apply(lambda x: 1 if x in capitals else 0)
    user = user.rename(columns={"city": "city_capital"})

    # Remove unnecessary features
    user = user.drop(['os', 'source'], axis=1)

    # Convert numerical features to float32
    numeric_columns = user.select_dtypes(include=['float64', 'int64']).columns
    user[numeric_columns] = user[numeric_columns].astype('float32')

    # Merge 'user' df and master df from learning
    user = user.combine_first(data)

    # Convert numerical categorical to int32
    user['exp_group'] = user['exp_group'].astype('int32')

    print(user.shape)
    print(user.main_topic_liked.isna().sum())
    print(user.user_id.nunique())
    print(user.post_id.nunique())
    print(data.user_id.nunique())

    return user

# Send .csv file to PostgreSQL using chunks and fast COPY method
def csv_to_sql(csv_path: str, table_name: str, type_map: dict, chunksize=5000, sep=";"):

    engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)
    metadata = MetaData()

    try:
        # Read .csv header
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(sep)

        # Create new SQL table
        columns_def = [Column(col, type_map.get(col, VARCHAR(255))) for col in header]
        table = Table(table_name, metadata, *columns_def)
        metadata.drop_all(engine, [table])
        metadata.create_all(engine)
        print(f"Table {table_name} recreated")

        # Calculate num of lines and chunks for tqdm progress bar
        total_lines = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        total_chunks = (total_lines + chunksize - 1) // chunksize

        # Open raw_connection: low-level approach
        conn = engine.raw_connection()
        cursor = conn.cursor()

        for chunk in tqdm(
            pd.read_csv(csv_path, sep=sep, chunksize=chunksize, encoding="utf-8"),
            total=total_chunks,
            desc="COPY chunks",
            unit="chunk"
        ):
            # Conbert types
            for col in chunk.columns:
                if col in type_map:
                    if type_map[col] == Integer:
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("int32")
                    elif type_map[col] == REAL:
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")
                    elif isinstance(type_map[col], VARCHAR):
                        chunk[col] = chunk[col].astype(str)

            # Ignore NaNs, save to buffer
            buffer = StringIO()
            chunk.to_csv(buffer, sep=sep, index=False, header=False)
            buffer.seek(0)

            # Copy buffer to the SQL table
            cursor.copy_expert(
                f"COPY {table_name} FROM STDIN WITH (FORMAT csv, DELIMITER '{sep}')",
                buffer
            )

            # Delete temp buffer, chunk and collect garbage
            buffer.close()
            del chunk
            gc.collect()

        # Finish connection
        conn.commit()
        cursor.close()
        conn.close()
        print("COPY completed successfully")

    finally:
        engine.dispose()

# Load big DF from the DB using chunks - ORM approach
def load_features(type_map:dict) -> pd.DataFrame:
    engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)
    chunksize = int(os.getenv('CHUNKSIZE'))
    chunks = []

    table_name = os.getenv('FEATURES_DF_NAME')

    try:
        print(f"from sql - start loading {table_name}")

        #
        with engine.connect() as conn:
            iterator = pd.read_sql(table_name, conn, chunksize=chunksize)
            for chunk in iterator:
                # Convert types by TYPE_MAP in the chunk
                for col, col_type in type_map.items():
                    if col in chunk.columns:
                        if col_type == Integer:
                            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('int32')
                        elif col_type == VARCHAR(50):
                            chunk[col] = chunk[col].astype(str)
                        elif col_type == REAL:
                            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")

                chunks.append(chunk)

        print(f"from sql - {table_name} loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Data loading error: {e}")

    df = pd.concat(chunks, ignore_index=True)
    total_memory = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"\nMemory size for the downloaded df: {total_memory:.2f} MB")
    print(df.shape)
    return df