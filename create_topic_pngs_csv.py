# This file reads in the cleaned template data directory and creates
# csv file with topic and png pairs.
import os

topic_pngs_filename = 'topic_pngs.csv'
root_dir = './presentation_template_data_cleaned/'

print("Reading", root_dir)
with open(topic_pngs_filename, 'w') as outfile:
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        topic = dir_name.replace(root_dir, '')
        for filename in file_list:
            if filename[-4:] == '.png':
                outfile.write(f'{topic},{filename}\n')

print("Created", topic_pngs_filename)
