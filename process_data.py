import sys
import getopt
import os
from os.path import isfile, join
import re
import json
import csv
import xlrd
import numpy as np
from collections import OrderedDict
from scipy import stats


def extract_data_from_cvs(cvs_file):
    # Extract gait data from cvs file
    with open(cvs_file) as cf:
        csv_reader = csv.reader(cf, delimiter=',')
        gait_data = list()
        for i, row in enumerate(csv_reader):
            if i == 0:
                # Omit first row containing field names
                pass
            else:
                try:
                    # Extract row containing time, and (x, y, z) coordinates of 17 markers
                    gait_data.append([float(row[4])] + row[11:17] + row[20:65])
                except ValueError:
                    pass
    return np.array(gait_data, dtype=np.float64)


def down_sample_data(data):
    # Down sample frames to the range of ~15 fps
    rows_list = list()
    for i, row in enumerate(data):
        if i%2 == 1:
            rows_list.append(row)
    return np.array(rows_list, dtype=np.float64)


def clean_data(data_dict):
    # Clean gait data by standardizing frame rate ...
    for samp_id, v_dict in data_dict.items():
        for k, v in v_dict.items():
            median_time_diff = np.median(v[1:,0] - v[:-1,0])
            min_time_diff = np.min(v[1:,0] - v[:-1,0])
            max_time_diff = np.max(v[1:,0] - v[:-1,0])
            if median_time_diff < 0.06:
                # If median of gait fps is not ~15 fps
                data_dict[samp_id][k] = down_sample_data(v)

    return


def segment_data(data_dict, seg_size):
    # Segment gait data into data points of size 'seg_size' frames
    robot_subjects_dict = OrderedDict()
    for samp_id, v_dict in data_dict.items():
        robot_subjects_dict[samp_id] = OrderedDict()
        segments_list = list()
        for k, v in v_dict.items():
            num_rows = v.shape[0]
            # Exclude the top few frames
            pad = num_rows % seg_size
            # Unroll frames into a single long vector
            segments_list.append(np.array(np.reshape(v[pad:,1:], (-1, 51*seg_size))))

        robot_subjects_dict[samp_id]['X'] = np.vstack(segments_list).tolist()
    return robot_subjects_dict


def extract_labels(data_dict, labels_file):
    # Open xlsx file
    wb = xlrd.open_workbook(labels_file)
    labels_sheet = wb.sheet_by_index(0)
    num_rows = labels_sheet.nrows
    print("num_rows: ", num_rows)

    # Extract subject information (id, gender, height, weight, etc.) from labels file and
    # add it to the respective subject data dictionary
    for r in range(1, num_rows):
        row = labels_sheet.row_values(r)
        id = int(row[1]) - 1
        data_dict[id]['id'] = id
        data_dict[id]['age'] = int(row[3])
        data_dict[id]['gender'] = 'male' if row[5] == 'MASCULINO' else 'female'
        data_dict[id]['sex'] = 0 if row[5] == 'MASCULINO' else 1
        data_dict[id]['height'] = float(row[6])
        data_dict[id]['weight'] = float(row[7])

    return data_dict


def process_robot_data(robot_files, robot_out_file, seg_sec=1.0):
    robot_files_dict = OrderedDict()
    seg_size = int(15.0 * seg_sec)

    # Extract Subject id and gait id
    for f in robot_files:
        subject_id = int(f.split('_')[2]) - 1
        gait_id = int(f.split('_')[3].split('.')[0]) - 1
        try:
            robot_files_dict[subject_id][gait_id] = f
        except:
            files_dict = OrderedDict()
            robot_files_dict[subject_id] = files_dict
            robot_files_dict[subject_id][gait_id] = f

    robot_data_dict = OrderedDict()
    # Extract gait data from cvs files
    for r in range(len(robot_files_dict)):
        robot_data_dict[r] = OrderedDict()
        files_count = len(robot_files_dict[r])
        for f in range(files_count):
            robot_data_dict[r][f] = extract_data_from_cvs(robot_files_dict[r][f])

    # Clean gait data - Standardize frame rate
    clean_data(robot_data_dict)

    # Segment gait data into data points
    robot_subject_dict = segment_data(robot_data_dict, seg_size)

    # Extract subject information (id, gender, height, weight, etc.) from labels file
    robot_subject_dict = extract_labels(robot_subject_dict, 'data/PARTICIPANTS_INFORMATION.xlsx')

    # Print subject information and data size
    for k, v in robot_subject_dict.items():
        print("%d: X(%d, %d) - id: %d - age: %d - sex: %d - height: %f - weight: %f" %
              (k+1, len(v['X']), len(v['X'][0]), v['id']+1, v['age'], v['sex'], v['height'],
               v['weight']))

    # Dump extracted data to file in JSON format
    with open(robot_out_file, 'w') as fp:
        json.dump(robot_subject_dict, fp, indent=4)


def main():
    rawdata_path = 'data/raw_data'
    robot_out_file = 'data/robot_data_file.dat'
    _, _, filenames = next(os.walk(rawdata_path))

    robot_files = list()
    for f in filenames:
        if 'robot' in f:
            robot_files.append(join(rawdata_path, f))

    process_robot_data(robot_files, robot_out_file, seg_sec=1.0)


if __name__ == "__main__":
    main()