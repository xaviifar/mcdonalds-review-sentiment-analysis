import pandas as pd
import os

PATH = os.path.join(os.getcwd(), "./", "data")

dfs = [pd.read_csv(os.path.join(PATH, f), index_col=0)
    for f in os.listdir(PATH) if f.endswith(".csv")]

df = pd.concat(dfs, axis=0).sort_index()

columns_to_drop = ["place_id","place_name","review_link",'name',"reviewer_id",'reviewer_profile',
       'original_language','translated_language',
       'published_at', 'published_at_date', 'response_from_owner_text',
       'response_from_owner_ago', 'response_from_owner_date', 'response_from_owner_translated_text', 'avatar_link',
       'experience_details', 'review_photos', 'review_origin']

existing_cols = [c for c in columns_to_drop if c in df.columns]
df = df.drop(columns=existing_cols)

df.drop_duplicates(subset="review_id")

df = df.dropna(subset=["review_text", "review_translated_text"])

df["rating_bin"] = (df["rating"] >= 3).astype(int)

df.to_csv("./data/clean_dataset.csv")

