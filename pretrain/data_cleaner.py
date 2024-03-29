import os
import threading
import tarfile
from langdetect import detect
import textacy.preprocessing as tprep


def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False


def clean_text(text):

    text = tprep.normalize.whitespace(text)
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    text = tprep.remove.html_tags(text)
    text = tprep.remove.brackets(text,["square", "curly"])
    if len(text) > 250:
        return text


def process_directory(subdir, root_dir, excluded_dirs):
    if any(excluded_dir in subdir for excluded_dir in excluded_dirs):
        return

    english_files = []
    temp_files = []
    for file in os.listdir(subdir):
        if file.endswith('.txt'):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if is_english(content):
                cleaned_content = clean_text(content)
                cleaned_file_path = os.path.join(subdir, "cleaned_" + file)
                os.makedirs(os.path.dirname(cleaned_file_path), exist_ok=True)

                with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                english_files.append(cleaned_file_path)
                temp_files.append(cleaned_file_path)

    if english_files:
        parent_dir = os.path.basename(os.path.dirname(subdir))
        current_dir = os.path.basename(subdir)
        tar_name = f"{parent_dir}_{current_dir}.tar.gz"
        tar_path = os.path.join(subdir, tar_name)

        with tarfile.open(tar_path, "w:gz") as tar:
            for file_path in english_files:
                tar.add(file_path, arcname=os.path.basename(file_path))


        extract_dir = os.path.join(subdir, current_dir + "_extracted")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)


    for temp_file in temp_files:
        os.remove(temp_file)


def process_files(root_dir, excluded_dirs):
    threads = []
    for subdir, dirs, files in os.walk(root_dir):
        if not any(excluded_dir in subdir for excluded_dir in excluded_dirs):
            t = threading.Thread(target=process_directory, args=(subdir, root_dir, excluded_dirs))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()



#root_dir = './testfile'

excluded_dirs = ['not', 'no2'] 
process_files(root_dir, excluded_dirs)
