#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can train the models consistently.    不要编辑这个脚本。更改将被丢弃，以便我们可以一致地训练模型。

# This file contains functions for training models for the Challenge. You can run it as follows:    这个文件包含了训练挑战模型的函数。你可以这样运行它：
#
#   python train_model.py data model
#
# where 'data' is a folder containing the Challenge data and 'model' is a folder for saving your model. ‘data’是一个包含挑战数据的文件夹，‘model’是一个用于保存模型的文件夹。

import sys
from helper_code import is_integer
from team_code import train_challenge_model

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    # Define the data and model foldes. 定义数据和模型文件夹。
    data_folder = sys.argv[1]
    model_folder = sys.argv[2]

    # Change the level of verbosity; helpful for debugging. 改变冗余级别；有助于调试。
    if len(sys.argv) == 4 and is_integer(sys.argv[3]):
        verbose = int(sys.argv[3])
    else:
        verbose = 1

    train_challenge_model(data_folder, model_folder, verbose)  ### Teams: Implement this function!!!    队伍：实现这个函数！！！
