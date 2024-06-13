import os
import shutil
import logging
import datetime

def create_exp_folder(base_folder):
    # 检查exp文件夹是否存在，如果不存在则创建
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # 遍历已存在的文件夹，获取最大的序号
    existing_folders = [name for name in os.listdir(base_folder) if name.startswith('exp')]
    if existing_folders:
        max_index = max(int(name[3:]) for name in existing_folders)
    else:
        max_index = 0

    # 创建新的文件夹
    new_folder_name = f'exp{max_index + 1}'
    new_folder_path = os.path.join(base_folder, new_folder_name)
    os.makedirs(new_folder_path)

    return new_folder_path


def folder_copy(dest_folder, model_name=None):
    train_file = os.path.abspath(__file__)
    train_name = os.path.basename(train_file)
    shutil.copy(train_file, os.path.join(dest_folder, train_name))
    shutil.copy('Train/utils.py',os.path.join(dest_folder, 'utils.py'))
    shutil.copy('main.py', os.path.join(dest_folder, 'main.py'))

    if model_name is not None:
        model_path = model_name + '.py'
        model_file = os.path.join('Models', model_path)
        model_name = os.path.basename(model_file)
        shutil.copy(model_file, os.path.join(dest_folder, model_name))


def pre_train(model_name='exp', model_path=None, train_log_name='training.log'):
    # copy the python file for model&train
    exp_folder = model_name
    exp_dest = create_exp_folder(exp_folder)
    folder_copy(exp_dest, model_path)
    # logging settings
    log_file_name = os.path.join(exp_dest, train_log_name)
    logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 设置控制台输出的日志级别
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

    current_time = datetime.datetime.now()
    logging.info(f'Training start at : {current_time}')
    return exp_dest
