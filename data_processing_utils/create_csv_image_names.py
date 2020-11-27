import pandas as pd
import glob
import os
import re

if __name__ == '__main__':
    img_name_list = []
    image_path = []
    for file in glob.glob('./full/*/*.png'):
        img_name_list.append(os.path.basename(file))
        image_path.append(file)

    df = pd.DataFrame({'image_path':image_path,'img_name_list':img_name_list})

    df['number_key'] = df[['img_name_list']].applymap(lambda x: re.findall('[0-9]+', x)[0])
    df['number_key'] = df['number_key'].astype('int64')
    print(df.info())
    ppt_desc_df = pd.read_csv('ppt_description.csv')
    ppt_desc_df['number_key'] = ppt_desc_df[['img_label']].applymap(lambda x: re.findall('[0-9]+', x)[0])
    ppt_desc_df['number_key'] = ppt_desc_df['number_key'].astype('int64')
    print(ppt_desc_df.info())
    img_desc_mapping = df.merge(ppt_desc_df, how = 'left', left_on='number_key',right_on='number_key')
    img_desc_mapping.dropna(inplace=True)
    img_desc_mapping.drop(['number_key'], axis=1, inplace=True)

    img_desc_mapping.to_csv('img_text_mapping.csv',index = False)

