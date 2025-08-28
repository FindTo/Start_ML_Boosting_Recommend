from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from loguru import logger
from database import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import desc
from table_post import Post
from table_user import User
from table_feed import Feed
from schema import UserGet, PostGet, FeedGet
from typing import List
from datetime import datetime
from get_features_table import get_user_features
from learn_model import get_post_df
from get_predict_by_model import load_models

# Load env variables
load_dotenv()

# Set of the model features with correct order
columns = [ 'topic',
            'cluster_1',
            'cluster_2',
            'cluster_3',
            'cluster_4',
            'cluster_5',
            'cluster_6',
            'cluster_7',
            'cluster_8',
            'cluster_9',
            'text_length',
            'gender',
            'age',
            'country',
            'exp_group',
            'city_capital',
            'post_likes',
            'post_views',
            'hour',
            'month',
            'day',
            'time_indicator',
            'main_topic_liked',
            'main_topic_viewed',
            'views_per_user',
            'likes_per_user']

user_df = get_user_features()

calc_features = ['main_topic_liked',
                 'main_topic_viewed',
                 'views_per_user',
                 'likes_per_user']
modes_calc_features = {col: user_df[col].mode(dropna=True)[0] for col in calc_features}

total_memory = user_df.memory_usage(deep=True).sum() / (1024**2)
print(f"\nMemory size for user_df: {total_memory:.2f} MB")

# Fill NaNs in user_df
modes = {col: user_df[col].mode(dropna=True)[0] for col in user_df.columns}
user_df = user_df.fillna(modes)

post_pool = user_df[[  'post_id',
                       'post_likes',
                       'post_views',
                       'text_length',
                       'cluster_1',
                       'cluster_2',
                       'cluster_3',
                       'cluster_4',
                       'cluster_5',
                       'cluster_6',
                       'cluster_7',
                       'cluster_8',
                       'cluster_9',
                       'topic'
                       ]].drop_duplicates('post_id').reset_index()

total_memory = post_pool.memory_usage(deep=True).sum() / (1024**2)
print(f"\nMemory size for post_pool: {total_memory:.2f} MB")

post_df = get_post_df()

total_memory = post_df.memory_usage(deep=True).sum() / (1024**2)
print(f"\nMemory size for post_df: {total_memory:.2f} MB")

model = load_models()

app = FastAPI()

def get_db():
    with SessionLocal() as db:
        return db

@app.get("/user/{id}", response_model = UserGet)
def get_user(id: int, db: Session = Depends(get_db)):

    data = db.query(User).filter(User.id == id).first()

    if data == None:

        raise HTTPException(404, "user not found")

    else:
        logger.info(data)
        return data

@app.get("/post/{id}", response_model = PostGet)
def get_post(id: int, db: Session = Depends(get_db)):

    data = db.query(Post).filter(Post.id == id).first()

    if data == None:

        raise HTTPException(404, "post not found")

    else:

        return data

@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):

    data = db.query(Feed).filter(Feed.user_id == id).order_by(desc(Feed.time)).limit(limit).all()
    logger.info(data)

    return data

@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: int = 10, db: Session = Depends(get_db)):

    return db.query(Feed).filter(Feed.post_id == id).order_by(desc(Feed.time)).limit(limit).all()

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:


    # Taking user data by ID from the original user table
    user_features = user_df.loc[user_df['user_id'] == id, ['user_id',
                                                           'gender',
                                                           'age',
                                                           'country',
                                                           'exp_group',
                                                           'city_capital',
                                                           'main_topic_liked',
                                                           'main_topic_viewed',
                                                           'views_per_user',
                                                           'likes_per_user']]

    # Fill empty spaces with modes
    user_features = user_features.fillna(modes_calc_features)

    try:
        # Use first row of the filtered by user_id dataframe
        user_features = user_features.iloc[0].to_frame().T.reset_index()

    except IndexError:
        # if incorrect user_id - 404 error
        raise HTTPException(404, detail=f"user {id} not found")


    # Get time features from timestamp
    user_features['hour'] = time.hour
    user_features['month'] = time.month
    user_features['day'] = time.day

    # Time indicator in hours from the beginning of 2021
    user_features['time_indicator'] = (time.year - 2021) * 360 * 24 + time.month * 30 * 24 + time.day * 24 + time.hour

    # Merge with user features line and preparing for prediction
    X = post_pool.copy()
    for col in user_features.columns:
        if col not in ['user_id', 'post_id']:
            X[col] = user_features[col].iloc[0]

    # Convert float64 numerical features to float32
    numeric_columns = X.select_dtypes(include=['float64']).columns
    X[numeric_columns] = X[numeric_columns].astype('float32')

    # Convert int64 numerical features to int32
    numeric_columns = X.select_dtypes(include=['int64']).columns
    X[numeric_columns] = X[numeric_columns].astype('int32')

    # Like probability prediction
    X['ax'] = model.predict_proba(X[columns],
                                  thread_count=-1)[:, 1]

    # First n=limit post with the highest like probability
    posts_recommend = (X.drop_duplicates('post_id')
                       .sort_values(by='ax', ascending=False)
                       .head(limit)['post_id']
                       .astype(float)
                       .astype(int)
                       .to_list()
                       )

    # Getting post data by obtained IDs from the original post_df
    posts_meta = (post_df.loc[post_df['post_id'].isin(posts_recommend),['post_id', 'text', 'topic']]
                  .drop_duplicates('post_id')
                  .set_index('post_id')
                  .reindex(posts_recommend))

    posts_recommend_list = [
        PostGet(id=int(pid), text=row['text'], topic=row['topic'])
        for pid, row in posts_meta.iterrows()
    ]

    return posts_recommend_list
