from utils_detection import *
import os

class DatasetLoader:
    def __init__(self, dataset_folder_location, sequence_length):
        self.X = None
        self.Y = None
        self.sequence_length = sequence_length
        self.dataset_folder_location = dataset_folder_location
        self.load_dataset()
        print(self.X.shape, self.Y.shape)


        

    def load_dataset(self):
        all_folders_input = os.listdir(self.dataset_folder_location)

        for folder_name in all_folders_input:
            current_location = f"{self.dataset_folder_location}\\{folder_name}\\"
            print(f"Loading data from folder {current_location}")

            files_name = os.listdir(current_location)
            files_names_distinct = self.remove_trailing_characters_file_names_return_set(files_name)

            y_value = DatasetLoader.encode_string(folder_name)

            for file in files_names_distinct:
                file_path = f"{current_location}\\{file}_"
                file_path_limbs_csv = f"{file_path}limbs.csv"
                file_path_yolo_csv = f"{file_path}yolo.csv"

                if os.path.exists(file_path_limbs_csv) == False or os.path.exists(file_path_yolo_csv) == False:
                    print(f"Couldn't find one of the files {file_path_limbs_csv} or {file_path_yolo_csv}, quitting")
                    return

                detections_made_human = load_from_csv_limbs(file_path_limbs_csv)
                yolo_detections = load_from_csv_YOLO(file_path_yolo_csv)

                matrix_inverse = convert_2D_Human(detections_made_human)

                matrice_norm = normalize_detection_limbs(yolo_detections, matrix_inverse)
                matrix_format_normalized = convert_to_2D_matrix(matrice_norm)
                sequences, y_vector = get_all_sequences_from_2D_format(matrix_format_normalized, self.sequence_length, y_value) #trebuie cea normalizata

                if self.X is None:
                    self.X = sequences
                    self.Y = y_vector
                else:
                    self.X = np.append(self.X, sequences, axis = 0)
                    self.Y = np.append(self.Y, y_vector)


    def remove_trailing_characters_file_names_return_set(self, files_names):
        for i, file_name in enumerate(files_names):
                #removing the trailing characters in the name
                files_names[i] = files_names[i].rpartition("_")[0]

        return set(files_names)


    def encode_string(string):
        if string == 'backhand':
            return 1
        if string == 'backhand_slice':
            return 2
        if string == 'backhand_volley':
            return 3
        if string == 'backhand2hands':
            return 4
        if string == 'flat_service':
            return 5
        if string == 'forehand_flat':
            return 6
        if string == 'forehand_openstands':
            return 7
        if string == 'forehand_slice':
            return 8
        if string == 'forehand_volley':
            return 9
        if string == 'kick_service':
            return 10
        if string == 'slice_service':
            return 11
        if string == 'smash':
            return 12
        return None
