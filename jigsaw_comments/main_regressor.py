# Build ridge cv with tf-idf
# Add multiple data sources. Use as test the comment_to_score.csv
# Run ridge over tfidf vectors and generate p1 and p2 for the validation set csv file 
# Predict for the comments_to_score for each fold and average. Finally, combine all predictions to generate final score.

import pandas as pd
import numpy as np
import os

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import rankdata

def ridge_cv(vec, X, y, X_test, folds, stratified, df_val):
    kf = StratifiedKFold(n_splits=folds,shuffle=True, random_state=123)
    val_scores = []
    rmse_scores = []
    X_less_toxics = []
    X_more_toxics = []

    preds = []
    for fold, (train_index,val_index) in enumerate(kf.split(X,stratified)):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        model = Ridge()
        model.fit(X_train, y_train)

        rmse_score = mean_squared_error(model.predict(X_val), y_val, squared = False) 
        rmse_scores.append(rmse_score)

        X_less_toxic = vec.transform(df_val['less_toxic'])
        X_more_toxic = vec.transform(df_val['more_toxic'])

        p1 = model.predict(X_less_toxic)
        p2 = model.predict(X_more_toxic)

        X_less_toxics.append(p1)
        X_more_toxics.append(p2)

        # Validation Accuracy
        val_acc = (p1 < p2).mean()
        val_scores.append(val_acc)

        pred = model.predict(X_test)
        preds.append(pred)

        print(f"FOLD:{fold}, rmse_fold:{rmse_score:.5f}, val_acc:{val_acc:.5f}")

    mean_val_acc = np.mean(val_scores)
    mean_rmse_score = np.mean(rmse_scores)

    p1 = np.mean(np.vstack(X_less_toxics), axis=0 )
    p2 = np.mean(np.vstack(X_more_toxics), axis=0 )

    val_acc = (p1< p2).mean()

    print(f"OOF: val_acc:{val_acc:.5f}, mean val_acc:{mean_val_acc:.5f}, mean rmse_score:{mean_rmse_score:.5f}")
    
    preds = np.mean( np.vstack(preds), axis=0 )
    
    return p1, p2, preds


def create_train(df):
    toxic = 1.0
    severe_toxic = 2.0
    obscene = 1.0
    threat = 1.0
    insult = 1.0
    identity_hate = 2.0

    df['y'] = df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].max(axis=1)
    df['y'] = df["y"]+df['severe_toxic']*severe_toxic
    df['y'] = df["y"]+df['obscene']*obscene
    df['y'] = df["y"]+df['threat']*threat
    df['y'] = df["y"]+df['insult']*insult
    df['y'] = df["y"]+df['identity_hate']*identity_hate
    
    
    
    df = df[['comment_text', 'y', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].rename(columns={'comment_text': 'text'})

    #undersample non toxic comments  on Toxic Comment Classification Challenge

    min_len = (df['y'] >= 1).sum()
    df_y0_undersample = df[df['y'] == 0].sample(n=int(min_len*1.5),random_state=201)
    df = pd.concat([df[df['y'] >= 1], df_y0_undersample])
                                                
    return df


if __name__ == '__main__':

    REMOTE = True
    DEBUG = False

    if REMOTE:
        path = '/home/daca/kaggle_challenges/jigsaw_comments/data/'
    else:
        path = 'data/'

    df_val = pd.read_csv(os.path.join(path, "validation_data.csv"))
    df_test = pd.read_csv(os.path.join(path, "comments_to_score.csv"))
    juc_df = pd.read_csv(os.path.join(path, "all_data.csv"))
    rud_df = pd.read_csv(os.path.join(path, "ruddit_with_text.csv"))
    jc_train_df = pd.read_csv(os.path.join(path, "jigsaw-toxic-comment-train.csv"))
    jc_test_df = pd.read_csv(os.path.join(path, "test.csv"))
    temp_df = pd.read_csv(os.path.join(path, "test_labels.csv"))

    features = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

    if DEBUG:
        juc_df, jc_train_df = juc_df.sample(n=1000, random_state=1), jc_train_df.sample(n=1000, random_state=1)
        rud_df, jc_test_df = rud_df.sample(n=1000, random_state=1), jc_test_df.sample(n=1000, random_state=1)

    
    print(f"Train: {jc_train_df.shape[0]}")
    jc_test_df = jc_test_df.merge(temp_df, on ="id")

    #drop test data not used for scoring
    jc_test_df = jc_test_df.query ("toxic != -1")
    print(f"Test: {jc_test_df.shape[0]}")

    jc_df = jc_train_df.append ( jc_test_df ) 
    print(f"Train+Test:{jc_df.shape[0]}")

    jc_df.head()

    print(f'duplicated by text:{jc_df.duplicated("comment_text").sum()}')

    jc_df["toxic_subtype_sum"]=jc_df[features].sum(axis=1)
    jc_df["toxic_behaviour"]=jc_df["toxic_subtype_sum"].map(lambda x: x > 0)

    tot_toxic_behaviour = jc_df["toxic_behaviour"].sum()
    print(f'comments with toxic behaviour:{tot_toxic_behaviour}')


    df = jc_df.query("toxic_subtype_sum > 0").groupby(features).agg({"id":"count"}).reset_index().sort_values(by="id", ascending=False).head(10)
    df = df.rename (columns={"id":"count"})
    df["perc"] = df["count"]/tot_toxic_behaviour
    df.head(10)


    df = jc_df.query ("toxic == 1 or severe_toxic == 1").groupby(["toxic","severe_toxic"]).agg({"id":"count"}).reset_index()
    df = df.rename (columns={"id":"count"})
    df["perc"] = df["count"]/tot_toxic_behaviour
    df

    df = jc_df.query ("toxic == 0 and toxic_subtype_sum > 0" ).groupby(features).agg({"id":"count"}).reset_index()
    df = df.rename (columns={"id":"count"})
    df["perc"] = df["count"]/tot_toxic_behaviour
    df.sort_values(by="count", ascending=False)
    
    jc_train_df = create_train(jc_train_df)
    jc_test_df = create_train(jc_test_df)

                        
    jc_df = jc_train_df.append(jc_test_df)                           


    FOLDS = 5

    vec = TfidfVectorizer(analyzer='char_wb', max_df=0.5, min_df=3, ngram_range=(4, 6) )
    X = vec.fit_transform(jc_df['text'])
    y = jc_df["y"].values
    X_test = vec.transform(df_test['text'])

    stratified = np.around( y )
    jc_p1, jc_p2, jc_preds =  ridge_cv(vec, X, y, X_test, FOLDS, stratified, df_val=df_val)


    features = ["toxicity","severe_toxicity","obscene","insult","identity_attack", "sexual_explicit"]
    cols = ['id', 'comment_text', 'toxicity', 'severe_toxicity', 'obscene', 'threat','insult', 'identity_attack', 'sexual_explicit', 'toxicity_annotator_count']

    # Code for jigsaw-unintended-bias-in-toxicity

    print(f"jigsaw-toxic-comment-classification-challenge shape:{juc_df.shape[0]}")
    print(f'duplicated by id:{juc_df.duplicated("id").sum()}, duplicated by text:{juc_df.duplicated("comment_text").sum()}')

    juc_df[["id", "comment_text"] + features + ['toxicity_annotator_count']].head()

    juc_df = juc_df.query ("toxicity_annotator_count > 5")
    print(f"juc_df:{juc_df.shape}")

    juc_df['y'] = juc_df[[ 'severe_toxicity', 'obscene', 'sexual_explicit','identity_attack', 'insult', 'threat']].sum(axis=1)

    juc_df['y'] = juc_df.apply(lambda row: row["toxicity"] if row["toxicity"] <= 0.5 else row["y"] , axis=1)
    juc_df = juc_df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
    min_len = (juc_df['y'] > 0.5).sum()
    df_y0_undersample = juc_df[juc_df['y'] <= 0.5].sample(n=int(min_len*1.5),random_state=201)
    juc_df = pd.concat([juc_df[juc_df['y'] > 0.5], df_y0_undersample])


    FOLDS = 5
    vec = TfidfVectorizer(analyzer='char_wb', max_df=0.5, min_df=3, ngram_range=(4, 6) )
    X = vec.fit_transform(juc_df['text'])
    y = juc_df["y"].values
    X_test = vec.transform(df_test['text'])

    stratified = (np.around( y, decimals = 1  )*10).astype(int)
    juc_p1, juc_p2, juc_preds = ridge_cv(vec, X, y, X_test, FOLDS, stratified, df_val=df_val)


    # Code for ruddit dataset
    rud_df.head()


    print(f"rud_df:{rud_df.shape}")
    rud_df['y'] = rud_df['offensiveness_score'].map(lambda x: 0.0 if x <=0 else x)
    rud_df = rud_df[['txt', 'y']].rename(columns={'txt': 'text'})
    min_len = (rud_df['y'] < 0.5).sum()


    FOLDS = 5
    vec = TfidfVectorizer(analyzer='char_wb', max_df=0.5, min_df=3, ngram_range=(4, 6) )
    X = vec.fit_transform(rud_df['text'])
    y = rud_df["y"].values
    X_test = vec.transform(df_test['text'])

    stratified = (np.around ( y, decimals = 1  )*10).astype(int)
    rud_p1, rud_p2, rud_preds =  ridge_cv(vec, X, y, X_test, FOLDS, stratified, df_val=df_val)

    jc_max = max(jc_p1.max() , jc_p2.max())
    juc_max = max(juc_p1.max() , juc_p2.max())
    rud_max = max(rud_p1.max() , rud_p2.max())


    p1 = jc_p1/jc_max + juc_p1/juc_max + rud_p1/rud_max
    p2 = jc_p2/jc_max + juc_p2/juc_max + rud_p2/rud_max

    val_acc = (p1< p2).mean()
    print(f"Ensemble: val_acc:{val_acc:.5f}")

    df_val['p1'] = p1
    df_val['p2'] = p2
    df_val['diff'] = np.abs(p2 - p1)
    df_val['correct'] = (p1 < p2).astype('int')
    ### Incorrect predictions with similar scores
    print(df_val[df_val.correct == 0].sort_values('diff', ascending=True).head(30))

    df_val.to_csv(os.path.join(path, "validation_csv_pred_regressor.csv"), index=False)


    score = jc_preds/jc_max + juc_preds/juc_max + rud_preds/rud_max  
    ## to enforce unique values on score
    df_test['score'] = rankdata(score, method='ordinal')

    df_test[['comment_id', 'score']].to_csv("/home/daca/kaggle_challenges/jigsaw_comments/output_ridge/submission.csv", index=False)

    df_test.head()
