#!/usr/bin/env python

# Load libraries.
import os, sys, shutil, argparse


# Parse arguments.  解析参数。
def get_parser():
    description = 'Remove labels from the dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser


# Run script.
def run(args):
    # Iterate over subfolders in the input folder.  遍历输入文件夹中的子文件夹。
    for root, folder_names, file_names in os.walk(args.input_folder):
        input_folder = root

        # Extract the subfolder of the input folder and use it for the output folder.   提取输入文件夹的子文件夹，并将其用于输出文件夹。
        x, y = root, ''
        while not os.path.samefile(args.input_folder, x):
            x, y = os.path.split(x)[0], os.path.join(os.path.split(x)[1], y)
        output_folder = os.path.join(args.output_folder, y)

        # Create the output folder if it does not already exist.    如果不存在，则创建输出文件夹。
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over each file in each subfolder. 遍历每个子文件夹中的每个文件。
        for file_name in file_names:
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)

            parent_folder = os.path.basename(input_folder)
            file_root, file_ext = os.path.splitext(file_name)

            # If the file does not have labels, then copy it as-is. 如果文件没有标签，则按原样复制它。
            if not (file_ext == '.txt' and file_root == parent_folder):
                shutil.copy2(input_file, output_file)
            # Otherwise, if the file does have labels, then remove the labels and copy it.  否则，如果文件有标签，则删除标签并复制它。
            else:
                with open(input_file, 'r') as f:
                    input_lines = f.readlines()

                output_lines = [l for l in input_lines if not (l.startswith('Outcome') or l.startswith('CPC'))]
                output_string = ''.join(output_lines)

                with open(output_file, 'w') as f:
                    f.write(output_string)


if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
