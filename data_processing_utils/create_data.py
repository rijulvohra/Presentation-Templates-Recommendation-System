import pandas as pd
import os
import re

if __name__ == '__main__':
    train_df = pd.read_csv('./recommender_data/train.csv')
    test_df = pd.read_csv('./recommender_data/test.csv')
    #val_df = pd.read_csv('./recommender_data/val.csv')
    desc_df = pd.read_csv('./ppt_description.csv')

    print(train_df.shape)
    print(test_df.shape)
    #print(val_df.shape)
    print(desc_df.head())
    #full_df = pd.concat([train_df,test_df])
    #full_df['key'] = full_df[['image_label']].applymap(lambda x: x.strip('.png'))
    #full_df['number_key'] = full_df[['image_label']].applymap(lambda x: re.findall(r'\b\d+\b',x))
    #full_df['number_key'] = full_df[['image_label']].applymap(lambda x: re.findall('[0-9]+', x)[0])
    train_df['number_key'] = train_df[['image_label']].applymap(lambda x: re.findall('[0-9]+',x)[0])
    test_df['number_key'] = test_df[['image_label']].applymap(lambda x: re.findall('[0-9]+',x)[0])
    desc_df['number_key'] = desc_df[['img_label']].applymap(lambda x: re.findall('[0-9]+', x)[0])
    train_merge_df = train_df.merge(desc_df,how='left',left_on='number_key',right_on='number_key')
    test_merge_df = test_df.merge(desc_df,how='left',left_on='number_key',right_on='number_key')
    train_merge_df.dropna(inplace = True)
    test_merge_df.dropna(inplace = True)
    train_merge_df.drop(['img_label','number_key'],axis=1,inplace=True)
    test_merge_df.drop(['img_label','number_key'],axis=1,inplace=True)
    train_merge_df.to_csv('ppt_desc_train.csv',index = False)
    test_merge_df.to_csv('ppt_desc_test.csv',index = False)


