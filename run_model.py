#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can run the trained models consistently.  不要编辑这个脚本。更改将被丢弃，以便我们可以一致地运行训练过的模型。

# This file contains functions for running models for the Challenge. You can run it as follows: 这个文件包含了运行挑战模型的函数。你可以这样运行它：
#
#   python run_model.py models data outputs
#
# where 'models' is a folder containing the your trained models, 'data' is a folder containing the Challenge data,  其中“models”是一个包含您训练过的模型的文件夹，“data”是一个包含挑战数据的文件夹，
# and 'outputs' is a folder for saving your models' outputs.    而“outputs”是一个用于保存您的模型输出的文件夹。

import numpy as np, scipy as sp, os, sys
from helper_code import *
from team_code import load_challenge_models, run_challenge_models


# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, verbose):
    # Load model(s).
    if verbose >= 1:
        print('Loading the Challenge models...')

    # You can use this function to perform tasks, such as loading your models, that you only need to perform once.  您可以使用此函数执行一次性任务，例如加载您的模型。
    models = load_challenge_models(model_folder, verbose)  ### Teams: Implement this function!!!

    # Find the Challenge data.  找到挑战数据。
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise Exception('No data was provided.')

    # Create a folder for the Challenge outputs if it does not already exist.   如果不存在，则为挑战输出创建一个文件夹。
    os.makedirs(output_folder, exist_ok=True)

    # Run the team's model on the Challenge data.   在挑战数据上运行团队的模型。
    if verbose >= 1:
        print('Running the Challenge models on the Challenge data...')

    # Iterate over the patients.    遍历患者。
    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i + 1, num_patients))

        patient_id = patient_ids[i]

        # Allow or disallow the models to fail on parts of the data; this can be helpful for debugging. 允许或禁止模型在数据的某些部分失败；这对于调试是有帮助的。
        try:
            outcome_binary, outcome_probability, cpc = run_challenge_models(models, data_folder, patient_id,
                                                                            verbose)  ### Teams: Implement this function!!!   队伍：实现这个函数！！！
        except:
            if allow_failures:
                if verbose >= 2:
                    print('... failed.')
                outcome_binary, outcome_probability, cpc = float('nan'), float('nan'), float('nan')
            else:
                raise

        # Save Challenge outputs.

        # Create a folder for the Challenge outputs if it does not already exist.
        os.makedirs(os.path.join(output_folder, patient_id), exist_ok=True)
        output_file = os.path.join(output_folder, patient_id, patient_id + '.txt')
        save_challenge_outputs(output_file, patient_id, outcome_binary, outcome_probability, cpc)

    if verbose >= 1:
        print('Done.')


if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 4 or len(sys.argv) == 5):
        raise Exception(
            'Include the model, data, and output folders as arguments, e.g., python run_model.py model data outputs.')

    # Define the model, data, and output folders.
    model_folder = sys.argv[1]
    data_folder = sys.argv[2]
    output_folder = sys.argv[3]

    # Allow or disallow the model to fail on parts of the data; helpful for debugging.  允许或禁止模型在数据的某些部分失败；有助于调试。
    allow_failures = False

    # Change the level of verbosity; helpful for debugging. 改变冗余级别；有助于调试。
    if len(sys.argv) == 5 and is_integer(sys.argv[4]):
        verbose = int(sys.argv[4])
    else:
        verbose = 1

    run_model(model_folder, data_folder, output_folder, allow_failures, verbose)
