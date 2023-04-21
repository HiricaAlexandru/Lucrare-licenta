from utils_detection import *
import os
from torch.utils.data import Dataset
import torch
import random

class DatasetLoader(Dataset):
    def __init__(self, dataset_folder_location, sequence_length, device, test = False, test_files_dict = None):
        self.X = None
        self.Y = None
        self.sequence_length = sequence_length
        self.dataset_folder_location = dataset_folder_location
        self.device = device
        self.test = test
        self.test_files_dict = test_files_dict
        self.X_by_video = []
        self.Y_by_video = []
        self.load_dataset()
        
        self.X = torch.tensor(self.X).float().to(device)
        self.Y = torch.tensor(self.Y).long().to(device)


    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def load_dataset(self):
        all_folders_input = os.listdir(self.dataset_folder_location)

        for folder_name in all_folders_input:
            current_location = f"{self.dataset_folder_location}\\{folder_name}\\"
            print(f"Loading {'test' if self.test == True else 'training'} data from folder {current_location}")

            files_name = os.listdir(current_location)
            files_names_distinct = DatasetLoader.remove_trailing_characters_file_names_return_set(files_name)

            y_value = DatasetLoader.encode_string(folder_name)

            for file in files_names_distinct:
                file_path = f"{current_location}\\{file}_"
                file_path_limbs_csv = f"{file_path}limbs.csv"
                file_path_yolo_csv = f"{file_path}yolo.csv"

                if os.path.exists(file_path_limbs_csv) == False or os.path.exists(file_path_yolo_csv) == False:
                    print(f"Couldn't find one of the files {file_path_limbs_csv} or {file_path_yolo_csv}, quitting")
                    return
                
                if self.test == True and self.test_files_dict.get(file, False) == True:
                    #if we are selecting the training files
                    detections_made_human = load_from_csv_limbs(file_path_limbs_csv)
                    yolo_detections = load_from_csv_YOLO(file_path_yolo_csv)

                    matrix_inverse = convert_2D_Human(detections_made_human)

                    matrice_norm = normalize_detection_limbs(yolo_detections, matrix_inverse)
                    matrix_format_normalized = convert_to_2D_matrix(matrice_norm)
                    sequences, y_vector = get_all_sequences_from_2D_format(matrix_format_normalized, self.sequence_length, y_value) #trebuie cea normalizata

                    self.X_by_video.append(sequences)
                    self.Y_by_video.append(y_vector[0])
                    
                    if self.X is None:
                        self.X = sequences
                        self.Y = y_vector
                    else:
                        self.X = np.append(self.X, sequences, axis = 0)
                        self.Y = np.append(self.Y, y_vector)
                
                if self.test == False and self.test_files_dict.get(file, False) == False:
                    #if not
                    detections_made_human = load_from_csv_limbs(file_path_limbs_csv)
                    yolo_detections = load_from_csv_YOLO(file_path_yolo_csv)

                    matrix_inverse = convert_2D_Human(detections_made_human)

                    matrice_norm = normalize_detection_limbs(yolo_detections, matrix_inverse)
                    matrix_format_normalized = convert_to_2D_matrix(matrice_norm)
                    sequences, y_vector = get_all_sequences_from_2D_format(matrix_format_normalized, self.sequence_length, y_value) #trebuie cea normalizata
                    
                    self.X_by_video.append(sequences)
                    self.Y_by_video.append(y_vector[0])

                    if self.X is None:
                        self.X = sequences
                        self.Y = y_vector
                    else:
                        self.X = np.append(self.X, sequences, axis = 0)
                        self.Y = np.append(self.Y, y_vector)



    def remove_trailing_characters_file_names_return_set(files_names):
        for i, file_name in enumerate(files_names):
                #removing the trailing characters in the name
                files_names[i] = files_names[i].rpartition("_")[0]

        return set(files_names)


    def encode_string(string):
        if string == 'backhand':
            return 0
        if string == 'backhand_slice':
            return 1
        if string == 'backhand_volley':
            return 2
        if string == 'backhand2hands':
            return 3
        if string == 'flat_service':
            return 4
        if string == 'forehand_flat':
            return 5
        if string == 'forehand_openstands':
            return 6
        if string == 'forehand_slice':
            return 7
        if string == 'forehand_volley':
            return 8
        if string == 'kick_service':
            return 9
        if string == 'slice_service':
            return 10
        if string == 'smash':
            return 11
        return None
    
    def select_dataset_files_for_testing(dataset_location, test_percentage):
        all_folders_input = os.listdir(dataset_location)
        test_files = dict()

        for folder_name in all_folders_input:
            current_location = f"{dataset_location}\\{folder_name}\\"

            files_name = os.listdir(current_location)
            files_names_distinct = list(DatasetLoader.remove_trailing_characters_file_names_return_set(files_name))

            selected_indexes = random.sample(range(0, len(files_names_distinct)), len(files_names_distinct) // test_percentage)

            for i in selected_indexes:
                test_files[files_names_distinct[i]] = True

        return test_files

    def save_test_to_file(file_name, dictionary):
        f = open(file_name, 'w')
        for file_name in dictionary.keys():
            f.writelines(file_name + '\n')
        f.close()

    def load_test_from_file(file_name):
        f = open(file_name, 'r')
        test_files = dict()
        read_line = f.readline()

        while not(" " in read_line) and len(read_line) != 0:
            read_line = read_line.rstrip('\n')
            test_files[read_line] = True
            read_line = f.readline()
        
        f.close()
        return test_files


