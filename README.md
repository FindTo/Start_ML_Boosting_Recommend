Karpov Courses StartML Project, classic machine learning module.

https://karpov.courses/ml-start

---

# Introduction

It's the web service to get post recommendations for users in social network.
Retrieving data from tables User, Posts and Feed, the list of top recommended posts per selected user can be obtained 
by http request.

This service relies on a gradient boosting model (Catboost) to make recommendations. 

---

# Input SQL tables structure

## User_data

- age - User age (in profile)
- city - User city (in profile)
- country - User country (in profile)
- exp_group - Experimental group: some encrypted category
- gender - User's gender
- user_id - Unique user ID
- os - Operating system of the device used to access the social network
- source - Whether the user came to the app from organic traffic or from advertising

##  Post_text_df 

- post_id - Unique post identifier
- text - Text content of the post
- topic - Main topic

##  Feed_data 

- timestamp - The time when the view was made.
- user_id - The ID of the user who made the view.
- post_id - The ID of the viewed post.
- action - The type of action: view or like.
- target - 1 for views if a like was made almost immediately after the view, otherwise 0. The value is omitted for 
like actions.

---

# Modules overview 

**Python ver. 3.12**

- There is a dockerfile and .dockerignore inside the project, which are aimed for server operation. Docker version 
is supposed for web service demonstration at https://startmlboostingrecommend-production.up.railway.app/, where you can 
try to send query and receive a response.

The **main_script.py** contains all feature preparation functions and calls, you can launch it cnd check process. 
But .env file will be necessary for DB connections and output features file naming.
The basic pipelane: gradient boosting is being larned on merged dataframe, with features from 
Feed_data/User_data/Post_text_df. Some additional features are being produced also. The obtained .cbm model filed is 
located locally in `/sources` folder, the master dataframe from learning sent to the SQL database. There is a function 
check_model_locally() for local test purposes - check local hitrate using feed_data.


The **app.py** is used for web service operation. At initialization the master dataframe is being downloaded for user and 
post features dataframes preparation. In `/post/recommendations/` endpoint user features ore being prepared by user_id, 
timestamp, preloaded dataframes, and list of predictions is being made, with sorting and getting the most relevant ones. 
Activation using `uvicorn app:app --port 8000`, or using another port. Example of http query (GET method): 
http://startmlboostingrecommend-production.up.railway.app/post/recommendations/?id=121245&time=2021-01-06 12:41:55

File **learn_model.py** contains classes and functions for Catboost gradient boosting model training, including datasets 
preparations. That all is used in **main_script.py**.

Module **get_model.py** is only aimed for .cbm file downloading on remote or locally.

File **get_features_table.py** contains functions for user features df preparation from the master dataframe and functions
for sending/loading dataframes to/from the SQL database.

Some modules - like **database.py**, **schema.py**, **table_feed.py**, **table_post.py** and **table_user.py** are used 
for setting SQLAlchemy ORM and Pydantic data formats.

# Endpoints

- **/user/{id}**: find relevant user info by id={id} and return as JSON
- **/post/{id}**: find relevant post info  by id={id} and return as JSON
- **/user/{id}/feed/?limit=LIMIT**: should return all actions from the feed for the user with id = {id}, sorted by 
actuality with limit=LIMIT
- **/post/{id}/feed/?limit=LIMIT**: should return all actions from the feed for the post with id = {id}, sorted by 
actuality with limit=LIMIT
- **/post/recommendations/?id=ID&time=TIMESTAMP&limit=LIMIT**: should return the top limit=LIMIT of recommended posts for the user with
user_id=ID at moment of time=TIMESTAMP (string datetime format %Y-%m-%d %H:%M:%S). There is the core endpoint for
recommendations.




