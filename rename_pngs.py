import os

rootDir = './presentation_template_data'
for dir_name, subdir_list, file_list in os.walk(rootDir):
    print('Found directory: %s' % dir_name)
    for new_name in file_list:
        original_name = new_name
        new_name = new_name.replace('-template-16x9.', '.')
        new_name = new_name.replace('_ppt.', '.')
        if new_name != original_name:
            print(f'Changing {original_name} to {new_name}')
            os.rename(dir_name + '/' + original_name, dir_name + '/' + new_name)
