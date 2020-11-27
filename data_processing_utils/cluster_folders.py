# This file combines directory names from a given mapping
import os
import glob
from shutil import copyfile

v1_mapping_strings = [
    "nature leaves flowers fire buterflies tree animals rainbow sun sunset water stone-objects snow-objects",
    "agriculture",
    "art",
    "education",
    "engineering industrial",
    "games",
    "love-hearts-couples wedding-ppt-by-topics",
    "ink pencil poster",
    "sports sports-2",
    "technology",
    "astrology-astronomy stars",
    "history literature",
    "science",
    "transportation",
    "music",
    "ecology-recycle",
    "food-drinks-ppt",
    "business",
    "content-marketing marketing",
    "holidays-in-the-year fathers-day human-race-day womens-day childrens-day saint-valentines-day mothers-day new-year presidents-day thanksgiving-day"
]

v2_mapping_strings = [
    "nature leaves flowers fire buterflies tree animals rainbow sun sunset water stone-objects snow-objects agriculture ecology-recycle",
    "art",
    "education science",
    "engineering industrial",
    "games",
    "sports sports-2",
    "technology",
    "astrology-astronomy stars",
    "history literature",
    "transportation",
    "music"
]


def cluster_folders(old_root_dir, new_root_dir, mapping):
    print(f"Creating directory {new_root_dir}\n")
    os.mkdir(new_root_dir)

    # Loop through all the entries in the mappings list of lists and copy any files to the subdirectory
    # which is the first entry in the sublist
    for topics in mapping:
        clustered_topic = topics[0]
        new_subdir = new_root_dir + '/' + clustered_topic
        print("\nCreating subdir", new_subdir)
        os.mkdir(new_subdir)
        for topic in topics:
            # If the topic exists in the old subdir then copy all the files to the new subdir
            old_subdir = old_root_dir + '/' + topic
            if os.path.isdir(old_subdir):
                print(f"copying files from {old_subdir} to {new_subdir}")
                for old_png_file in glob.glob(old_subdir + "/*.png"):
                    new_png_file = old_png_file.replace(old_subdir, new_subdir)
                    print(f'Copying {old_png_file} to {new_png_file}')
                    copyfile(old_png_file, new_png_file)


def main():
    old_root_dir = "presentation_template_data_cleaned"

    new_root_dir_v1 = "presentation_template_data_v1"
    v1_mapping = [s.split(" ") for s in v1_mapping_strings]
    cluster_folders(old_root_dir, new_root_dir_v1, v1_mapping)
    print("\nFinished clustering into ", new_root_dir_v1)

    new_root_dir_v2 = "presentation_template_data_v2"
    v2_mapping = [s.split(" ") for s in v2_mapping_strings]
    cluster_folders(old_root_dir, new_root_dir_v2, v2_mapping)
    print("\nFinished clustering into ", new_root_dir_v2)


if __name__ == '__main__':
    main()
