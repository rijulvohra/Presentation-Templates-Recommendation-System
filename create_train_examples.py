# This file reads in the topic png pairs and the training text to create
# positive and negative testing data.
import random
import csv


def read_topic_pngs_dict(topic_pngs_filename):
    topic_pngs_dict = {}
    with open(topic_pngs_filename, mode='r', encoding='utf-8') as labels_file:
        for line in labels_file:
            topic, png_name = line.strip().split(',')
            if topic in topic_pngs_dict:
                topic_pngs_dict[topic].append(png_name)
            else:
                topic_pngs_dict[topic] = [png_name]
    return topic_pngs_dict


def main():
    topic_pngs_filename = 'topic_pngs.csv'
    train_text_filename = 'train_text.csv'
    train_examples_filename = 'train_examples.csv'
    topic_pngs_dict = read_topic_pngs_dict(topic_pngs_filename)

    # For each line in the training text file create a positive and negative random training sample.
    with open(train_text_filename, mode='r', encoding='utf-8') as train_text_file, \
            open(train_examples_filename, 'w') as output_file:
        csv_reader = csv.reader(train_text_file)
        header = next(csv_reader)  # skip the heading
        assert (header == ['text', 'topic', 'topic_label'])

        output_file.write('text, topic, image label, match\n')
        topics = list(topic_pngs_dict.keys())
        missing_topics = set()
        for text_line in csv_reader:
            topic = text_line[1]
            if topic not in topic_pngs_dict:
                if topic not in missing_topics:
                    print(topic, "does not have any associated .pngs")
                    missing_topics.add(topic)
            else:
                postive_png_name = random.choice(topic_pngs_dict[topic])
                output_file.write('"{}",{},{},1\n'.format(text_line[0], topic, postive_png_name))

                # Get a negative image
                negative_topic = random.choice(topics)
                while (negative_topic == topic):
                    negative_topic = random.choice(list(topic_pngs_dict))
                negative_png_name = random.choice(topic_pngs_dict[negative_topic])
                output_file.write('"{}",{},{},0\n'.format(text_line[0], negative_topic, negative_png_name))

    print("Finished writing", train_examples_filename)


if __name__ == '__main__':
    main()
