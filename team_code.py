#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,   编辑这个脚本以添加你的团队代码。有些函数是必需的，但你可以编辑大多数部分的必需函数，
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.  可选的库和函数。你可以改变或删除它们。
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import torch.nn

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
# 必需的函数。编辑这些函数以添加你的代码，但不要改变函数的参数。
#
################################################################################
from helper_code import *
import numpy as np
import os, numpy as np, scipy as sp, scipy.io
import torch.nn


# 定义一个神经网络，输入是1*6000的数据，输出是1*1的数据
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义神经网络的每层结构形式
        # 定义一个一维卷积层，卷积核大小1*30，步长1，输入通道数1，输出通道数1
        self.conv1 = torch.nn.Conv1d(1, 1, 30, 1)  # sride=1说明步长为1
        # 激活函数
        self.relu = torch.nn.ReLU()
        # 定义一个一维池化层，池化核大小1*5，步长5
        self.pool1 = torch.nn.MaxPool1d(1, 5)
        # 定义一个一维卷积层，卷积核大小1*30，步长1，输入通道数1，输出通道数1
        self.conv2 = torch.nn.Conv1d(1, 1, 30, 1)
        # 激活函数
        self.relu = torch.nn.ReLU()
        # 定义一个一维池化层，池化核大小1*5，步长5
        self.pool2 = torch.nn.MaxPool1d(1, 5)

        # 定义一个LSTM层，输入向量维度1，隐藏层维度1，2层LSTM
        # self.lstm = torch.nn.LSTM(234, 1, 2)    # 234=1*234
        # 定义一个全连接层，输入维度10，输出维度5
        self.out = torch.nn.Linear(234, 5)  # 输出维度为5
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # 卷积->激活->池化
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        # 卷积->激活->池化
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        # LSTM层
        # x = x.view(-1, 1, 234)
        # x, (h_n, h_c) = self.lstm(x, None)
        # 全连接层
        x = self.out(x)
        x = self.softmax(x)
        return x


# 定义一个训练函数
def train(data, label, net):
    # 定义神经网络的输入数据和标签
    x = torch.tensor(data).float()
    y = torch.tensor(label).float()
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 训练网络
    for epoch in range(50):
        # 输入数据通过神经网络
        out = net(x)
        # 计算损失函数
        loss = criterion(out, y)
        # 清空上一步的残余更新参数值
        optimizer.zero_grad()
        # 误差反向传播，计算参数更新值
        loss.backward()
        # 将参数更新值施加到net的parmeters上
        optimizer.step()

def split_and_test(data):
    data6000 = []
    prediclist = []
    data = np.array(data)


    for i in range(18):
        datak = data[i]
        data6000_1 = []
        data6000_2 = []
        data6000_3 = []
        data6000_4 = []
        data6000_5 = []
        for j in range(5):
            data1 = datak[j * 6000:(j + 1) * 6000]
            if j == 0:
                data6000_1.append(data1)
            elif j == 1:
                data6000_2.append(data1)
            elif j == 2:
                data6000_3.append(data1)
            elif j == 3:
                data6000_4.append(data1)
            elif j == 4:
                data6000_5.append(data1)

        predict1 = predic(data6000_1, Net1)  # 预测结果类型为tensor
        predict2 = predic(data6000_2, Net2)
        predict3 = predic(data6000_3, Net3)
        predict4 = predic(data6000_4, Net4)
        predict5 = predic(data6000_5, Net5)
        # 把预测结果的众数作为最终的预测结果
        predict = np.array([predict1, predict2, predict3, predict4, predict5])
        predict = np.concatenate(predict)
        predict = np.argmax(np.bincount(predict))
        prediclist.append(predict)
        # print(predict, label)

    # 把预测结果的众数作为最终的预测结果
    # print(prediclist)
    predict = np.argmax(np.bincount(prediclist))
    # print(predict)
    return prediclist
# 定义一个验证函数，测试其准确率
def predic(data,  net):
    # 定义神经网络的输入数据和标签
    x = torch.tensor(data).float()
    # 输入数据通过神经网络
    out = net(x)
    # 输出最大值的索引
    p = torch.max(out, 1)[1]
    # print(p.numpy().astype(int))

    return p.numpy().astype(int)


# 定义一个函数，接收18*30000的数据，先分成1*30000的数据，再分成1*6000的数据,所有数据全部输出
def split_and_train(data, label):
    data6000 = []
    prediclist = []
    data = np.array(data)

    # 把label转换成one-hot编码
    label = np.eye(5)[label]

    # print('标签', label)
    for i in range(18):
        datak = data[i]
        data6000_1 = []
        data6000_2 = []
        data6000_3 = []
        data6000_4 = []
        data6000_5 = []
        for j in range(5):
            data1 = datak[j * 6000:(j + 1) * 6000]
            if j == 0:
                data6000_1.append(data1)
            elif j == 1:
                data6000_2.append(data1)
            elif j == 2:
                data6000_3.append(data1)
            elif j == 3:
                data6000_4.append(data1)
            elif j == 4:
                data6000_5.append(data1)
        # print("到1了")
        train(data6000_1, label, Net1)
        # print("到2了")
        train(data6000_2, label, Net2)
        # print("到3了")
        train(data6000_3, label, Net3)
        # print("到4了")
        train(data6000_4, label, Net4)
        # print("到5了")
        train(data6000_5, label, Net5)
        # print("到6了")




def load_challenge_data_1(data_folder, patient_id):
    # Define file location. 定义文件位置。
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')

    # Load non-recording data.  加载非记录数据。
    patient_metadata = load_text_file(patient_metadata_file)
    recording_metadata = load_text_file(recording_metadata_file)

    # Load recordings.  加载录音。
    recordings = list()
    recording_ids = get_recording_ids(recording_metadata)
    recordingcount = 0
    for recording_id in recording_ids:
        if not is_nan(recording_id):  # Skip the hidden test data.  跳过隐藏的测试数据。
            recording_location = os.path.join(data_folder, patient_id, recording_id)  # 定义文件位置。
            recording_data, sampling_frequency, channels = load_recording(recording_location)  # 加载录音。
            recordingcount += 1
        else:
            continue
        recordings.append(recording_data)

    return patient_metadata, recording_metadata, recordings, recordingcount


def judge(num):
    if 0 <= num <= 1:
        return 0
    elif 2 <= num <= 4:
        return 1
    else:
        return 'NaN'


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    global Net1
    global Net2
    global Net3
    global Net4
    global Net5
    Net1 = Net()
    Net2 = Net()
    Net3 = Net()
    Net4 = Net()
    Net5 = Net()

    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.   如果不存在，则为模型创建一个文件夹。
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.  提取特征和标签。
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')  # 从挑战数据中提取特征和标签...



    recordingslist = []
    labellist = []
    recordingscountlist = []
    count = 0

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i + 1, num_patients))

        # Load data.    载入数据。
        patient_id = patient_ids[i]
        count += 1
        a, _, c, d = load_challenge_data_1(data_folder, patient_id)
        recordingscountlist.append(d)
        cpcnum = int(a[-2:-1]) - 1

        # print(cpcnum)
        for k in range(len(c)):
            labellist.append(cpcnum)

        recordingslist.extend(c)

    X_train, y_train = recordingslist, labellist


    for i in range(len(X_train)):
        split_and_train(X_train[i], y_train[i])



    # Save the models.
    save_challenge_model(model_folder, Net1, Net2, Net3, Net4, Net5)

    if verbose >= 1:
        print('Done.')


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models_test.sav')
    return joblib.load(filename)


# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.    运行您的训练模型。这个函数是必需的。您应该编辑这个函数来添加您的代码，但不要改变这个函数的参数。
def run_challenge_models(models, data_folder, patient_id, verbose):

    global Net1
    global Net2
    global Net3
    global Net4
    global Net5

    Net1= models['Net1']
    Net2 = models['Net2']
    Net3 = models['Net3']
    Net4 = models['Net4']
    Net5 = models['Net5']



    recordingslist = []
    count = 0
    patient_id
    a, _, c, d = load_challenge_data_1(data_folder, patient_id)
    recordingslist.extend(c)



    X_train = recordingslist

    kindcount = [0, 0, 0, 0, 0]

    for i in range(len(X_train)):
        voting18 = split_and_test(X_train[i])
        for k in voting18:
            kindcount[k] += 1


    # 统计kindcount中的最大值的索引
    maxindex = 0
    maxnum = kindcount[0]
    for j in range(0, 5):
        if kindcount[j] > maxnum:
            maxindex = j
            maxnum = kindcount[j]

    P = maxnum / sum(kindcount)
    outcome=judge(maxindex)


    return outcome, P, maxindex+1


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.    可选函数。您可以更改或删除这些函数和/或添加新函数。
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, Net1,Net2,Net3,Net4,Net5):
    d = {'Net1': Net1, 'Net2':Net2,'Net3':Net3,'Net4':Net4,'Net5':Net5}    # Create a dictionary of the models.  创建模型的字典。
    filename = os.path.join(model_folder, 'models_test.sav') # Save the models.  保存模型。
    joblib.dump(d, filename, protocol=0)    # Use protocol 0 for compatibility with Python 2.   使用协议0与Python 2兼容。


# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.   从患者元数据中提取特征。
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables  使用一位有效编码进行性别；添加更多变量
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male = 0
        other = 0
    elif sex == 'Male':
        female = 0
        male = 1
        other = 0
    else:
        female = 0
        male = 0
        other = 1

    # Combine the patient features. 组合患者特征。
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.    从记录数据和元数据中提取特征。
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.  计算每个记录的每个通道的平均值和标准差。
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels,
                                                     channels)  # Reorder the channels in the signal data, as needed,
            # for consistency across different recordings.  将信号数据中的通道重新排序，如有必要，以便在不同记录之间保持一致。
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std = float('nan') * np.ones(num_channels)

    # Compute the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of
    # the most recent recording.    计算最近记录的每个通道的δ、θ、α和β频带的功率谱密度。
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels,
                                                 channels)  # Reorder the channels in the signal data, as needed,
        # for consistency across different recordings.  将信号数据中的通道重新排序，如有必要，以便在不同记录之间保持一致。

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=0.5, fmax=8.0,
                                                          verbose=False)    # Compute the power spectral density for the delta frequency band.
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=4.0, fmax=8.0,
                                                          verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=8.0, fmax=12.0,
                                                          verbose=False)
        beta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0,
                                                         verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean = np.nanmean(beta_psd, axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack(
        (signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # Combine the features from the patient metadata and the recording data and metadata.   将患者元数据和记录数据和元数据的特征组合起来。
    features = np.hstack((patient_features, recording_features))

    return features
