#!/usr/bin/env python

# Load libraries.
import os, sys, shutil, argparse
from helper_code import *


# Parse arguments.  解析参数。
def get_parser():
    description = 'Truncate recordings for the provided hour limit.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-k', '--hour_limit', type=int, required=True)
    return parser


# Run script.   运行脚本。
def run(args):
    # Find the folders with the patient and recording metadata and data.    找到具有患者和记录元数据和数据的文件夹。
    patient_ids = find_data_folders(args.input_folder)

    # Iterate over each folder.  遍历每个文件夹。
    for patient_id in patient_ids:
        # Make output folder.   创建输出文件夹。
        os.makedirs(os.path.join(args.output_folder, patient_id), exist_ok=True)

        # Set and copy the patient metadata file.   设置并复制患者元数据文件。
        input_patient_metadata_file = os.path.join(args.input_folder, patient_id, patient_id + '.txt')
        output_patient_metadata_file = os.path.join(args.output_folder, patient_id, patient_id + '.txt')

        shutil.copy(input_patient_metadata_file, output_patient_metadata_file)

        # Set, truncate, and copy the recording metadata file.  设置，截断并复制记录元数据文件。
        input_recording_metadata_file = os.path.join(args.input_folder, patient_id, patient_id + '.tsv')
        output_recording_metadata_file = os.path.join(args.output_folder, patient_id, patient_id + '.tsv')

        input_recording_metadata = load_text_file(input_recording_metadata_file)
        hours = get_hours(input_recording_metadata)
        indices = [i for i, hour in enumerate(hours) if hour <= args.hour_limit]

        input_lines = input_recording_metadata.split('\n')
        lines = [input_lines[0]] + [input_lines[i + 1] for i in indices]
        output_recording_metadata = '\n'.join(lines)
        with open(output_recording_metadata_file, 'w') as f:
            f.write(output_recording_metadata)

        # Set and copy the recording data.  设置并复制记录数据。
        recording_ids = get_recording_ids(input_recording_metadata)
        for i in indices:
            recording_id = recording_ids[i]
            input_header_file = os.path.join(args.input_folder, patient_id, recording_id + '.hea')
            input_signal_file = os.path.join(args.input_folder, patient_id, recording_id + '.mat')
            output_header_file = os.path.join(args.output_folder, patient_id, recording_id + '.hea')
            output_signal_file = os.path.join(args.output_folder, patient_id, recording_id + '.mat')

            if os.path.isfile(input_header_file):
                shutil.copy(input_header_file, output_header_file)
            if os.path.isfile(input_signal_file):
                shutil.copy(input_signal_file, output_signal_file)


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
