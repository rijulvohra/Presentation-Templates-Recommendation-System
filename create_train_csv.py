import os

csv_filename = 'train_pngs.csv'
root_dir = './presentation_template_data/'

with open(csv_filename, 'w') as outfile:
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        category = dir_name.replace(root_dir, '')
        for filename in file_list:
            if filename[-4:] == '.png':
                outfile.write(f'{filename},{category}\n')
