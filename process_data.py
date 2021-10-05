import sys
import getopt
import os
from os.path import join
import json
import csv
import xlrd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


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
        if i % 2 == 1:
            rows_list.append(row)
    return np.array(rows_list, dtype=np.float64)


def clean_data(data_dict):
    # Clean gait data by standardizing frame rate ...
    for samp_id, v_dict in data_dict.items():
        for k, v in v_dict.items():
            median_time_diff = np.median(v[1:, 0] - v[:-1, 0])
            # min_time_diff = np.min(v[1:, 0] - v[:-1, 0])
            # max_time_diff = np.max(v[1:, 0] - v[:-1, 0])
            if median_time_diff < 0.06:
                # If median of gait fps is not ~15 fps
                data_dict[samp_id][k] = down_sample_data(v)

    return


def segment_data(data_dict, seg_size):
    # Segment gait data into data points of size 'seg_size' frames
    people_dict = OrderedDict()
    for samp_id, v_dict in data_dict.items():
        people_dict[samp_id] = OrderedDict()
        segments_list = list()
        for k, v in v_dict.items():
            num_rows = v.shape[0]
            # Exclude the top few frames
            pad = num_rows % seg_size
            # Unroll frames into a single long vector - Exclude the first column containing time
            segments_list.append(np.reshape(v[pad:, 1:], (-1, 51 * seg_size)).tolist())

        people_dict[samp_id]['X'] = segments_list
    return people_dict


def categorize_to_bins(data, n_bins='auto', plot=False, axs=None, x_label='Value'):
    # Catagorise data and create bins
    n, bins, patches = axs.hist(x=data, bins=n_bins, color='#0504aa', alpha=0.7, rwidth=0.85)

    if plot:
        axs.grid(axis='y', alpha=0.75)
        axs.set(xlabel=x_label)
        axs.set(ylabel='Frequency')

        # Set a clean upper y-axis limit.
        maxfreq = n.max()
        axs.set_ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    return bins, n


def get_bin_index(bins, value):
    num_bins = len(bins)
    idx = num_bins - 2  # By default assigned to the last bin

    for i in range(num_bins - 1):
        if value >= bins[i] and value < bins[i + 1]:
            idx = i
            break

    return int(idx)


def get_person_gender(id):
    labels_file = 'data/PARTICIPANTS_INFORMATION.xlsx'
    wb = xlrd.open_workbook(labels_file)
    labels_sheet = wb.sheet_by_index(0)
    num_rows = labels_sheet.nrows

    for r in range(1, num_rows):
        row = labels_sheet.row_values(r)
        if int(row[1]) - 1 == id:
            return 'male' if row[5] == 'MASCULINO' else 'female'

    return None


def extract_labels(data_dict, labels_file, male=True, female=True, n_bins='auto', plot=False):
    # Open xlsx file
    wb = xlrd.open_workbook(labels_file)
    labels_sheet = wb.sheet_by_index(0)
    num_rows = labels_sheet.nrows
    print("num_rows: ", num_rows)

    # Extract person height and weight information
    heights = list()
    weights = list()
    for r in range(1, num_rows):
        gender = 'male' if labels_sheet.cell_value(r, 5) == 'MASCULINO' else 'female'
        # Process data of the specified gender
        if (male and female) or (male and gender == 'male') or (female and gender == 'female'):
            heights.append(float(labels_sheet.cell_value(r, 6)))
            weights.append(float(labels_sheet.cell_value(r, 7)))

    fig, axs = plt.subplots(2, sharey=False, sharex=False)
    if plot:
        fig.suptitle('Height/Weight Distribution Histograms', fontsize=20)

    height_bins, height_bins_count = categorize_to_bins(heights, n_bins=n_bins, plot=plot,
                                                        axs=axs[0], x_label='Height')
    weight_bins, weight_bins_count = categorize_to_bins(weights, n_bins=n_bins, plot=plot,
                                                        axs=axs[1], x_label='Weight')

    print("height_bins:\n", height_bins)
    print("Height Min: %.1f  --  Max: %.1f" %
          (np.min(np.array(heights)), np.max(np.array(heights))))
    print("height_bins Diff:\n", (np.array(height_bins)[1:] - np.array(height_bins)[:-1]))
    print("\nweight_bins:\n", weight_bins)
    print("Weight Min: %.2f  --  Max: %.2f" %
          (np.min(np.array(weights)), np.max(np.array(weights))))
    print("weight_bins Diff:\n", (np.array(weight_bins)[1:] - np.array(weight_bins)[:-1]))
    print("\n")

    # Extract person information (id, gender, height, weight, etc.) from labels file and
    # add it to the respective person data dictionary
    for r in range(1, num_rows):
        row = labels_sheet.row_values(r)
        id = int(row[1]) - 1
        try:
            data_dict[id]['id'] = id
            data_dict[id]['age'] = int(row[3])
            data_dict[id]['gender'] = 'male' if row[5] == 'MASCULINO' else 'female'
            data_dict[id]['sex'] = 0 if row[5] == 'MASCULINO' else 1

            data_dict[id]['height'] = float(row[6])
            h_idx = get_bin_index(height_bins, data_dict[id]['height'])
            data_dict[id]['height_ctgry'] = h_idx
            data_dict[id]['height_bin'] = tuple((height_bins[h_idx], height_bins[h_idx + 1]))

            data_dict[id]['weight'] = float(row[7])
            w_idx = get_bin_index(weight_bins, data_dict[id]['weight'])
            data_dict[id]['weight_ctgry'] = w_idx
            data_dict[id]['weight_bin'] = tuple((weight_bins[w_idx], weight_bins[w_idx + 1]))
        except KeyError:
            pass

    print("height_bins_count:", height_bins_count)
    print("weight_bins_count:", weight_bins_count)
    print("\n")

    if plot:
        plt.show(block=False)
        plt.pause(0.01)
        input("Press Enter to continue...")

    return data_dict


def process_data(data_files, out_file, male=True, female=True, seg_sec=1.0, n_bins='auto',
                 plot=False):
    person_files_dict = OrderedDict()
    seg_size = int(15.0 * seg_sec)

    # Extract person id and gait id, and collect gait files per person
    for f in data_files:
        person_id = int(f.split('_')[2]) - 1
        gait_id = int(f.split('_')[3].split('.')[0]) - 1
        try:
            person_files_dict[person_id][gait_id] = f
        except KeyError:
            files_dict = OrderedDict()
            person_files_dict[person_id] = files_dict
            person_files_dict[person_id][gait_id] = f

    data_dict = OrderedDict()
    # Extract gait data from cvs files
    for r in range(len(person_files_dict)):
        gender = get_person_gender(r)
        # Process data of the specified gender
        if (male and female) or (male and gender == 'male') or (female and gender == 'female'):
            data_dict[r] = OrderedDict()
            files_count = len(person_files_dict[r])
            for f in range(files_count):
                data_dict[r][f] = extract_data_from_cvs(person_files_dict[r][f])

    # Clean gait data - Standardize frame rate
    clean_data(data_dict)

    # Segment gait data into data points
    person_dict = segment_data(data_dict, seg_size)

    # Extract person information (id, gender, height, weight, etc.) from labels file
    person_dict = extract_labels(person_dict, 'data/PARTICIPANTS_INFORMATION.xlsx', male=male,
                                 female=female, n_bins=n_bins, plot=plot)

    # Print person information and data size
    for k, v in person_dict.items():
        print(("%d: X(%d, %d) - id: %d - age: %d - sex: %d - [%d] height: %.2f (%.1f:%.1f) - " +
               "[%d] weight: %.2f (%.2f:%.2f)") %
              (k + 1, len(v['X']), len(v['X'][0]), v['id'] + 1, v['age'], v['sex'],
               v['height_ctgry'], v['height'], v['height_bin'][0], v['height_bin'][1],
               v['weight_ctgry'], v['weight'], v['weight_bin'][0], v['weight_bin'][1]))

    # Dump extracted data to file in JSON format
    with open(out_file, 'w') as fp:
        json.dump(person_dict, fp, indent=4)


def main(argv):
    male = True
    female = True
    seg_sec = 1.0
    n_bins = 'auto'
    plot = False
    robot = False
    vicon = False

    rawdata_path = 'data/gait_files'
    robot_out_file = 'data/robot_data_file.dat'
    vicon_out_file = 'data/vicon_data_file.dat'
    try:
        opts, args = getopt.getopt(argv, "h fmprvs:b:", ["female", "male", "plot", "robot", "vicon",
                                                         "sec=", "bins="])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-f", "--female"):
            female = False
        elif opt in ("-m", "--male"):
            male = False
        elif opt in ("-p", "--plot"):
            plot = True
        elif opt in ("-r", "--robot"):
            robot = True
        elif opt in ("-v", "--vicon"):
            vicon = True
        elif opt in ("-s", "--sec"):
            seg_sec = float(arg)
        elif opt in ("-b", "--bins"):
            n_bins = int(arg)

    if not(robot or vicon):
        sys.exit("Error: Choose either robot or vicon data to process with flags [-r|--robot] " +
                 "or [-v|--vicon] respectively.")
    elif robot and vicon:
        sys.exit("Error: Choose only of the data sources with flags [-r|--robot] or [-v|--vicon] " +
                 "respectively.")

    if male and female:
        print("Processing data of both genders")
    elif male:
        print("Processing data of Males only!")
    elif female:
        print("Processing data of Females only!")
    else:
        sys.exit("Error: Remove atleast one of [-f|--female] or [-m|--male] flags.")

    if robot:
        # Extract robot file names
        _, _, filenames = next(os.walk(rawdata_path))
        robot_files = list()
        for f in filenames:
            if 'robot' in f:
                robot_files.append(join(rawdata_path, f))

        process_data(robot_files, robot_out_file, male=male, female=female, seg_sec=seg_sec,
                     n_bins=n_bins, plot=plot)
    elif vicon:
        # Extract vicon file names
        _, _, filenames = next(os.walk(rawdata_path))
        vicon_files = list()
        for f in filenames:
            if 'vicon' in f.lower():
                vicon_files.append(join(rawdata_path, f))

        process_data(vicon_files, vicon_out_file, male=male, female=female, seg_sec=seg_sec,
                     n_bins=n_bins, plot=plot)


if __name__ == "__main__":
    main(sys.argv[1:])
