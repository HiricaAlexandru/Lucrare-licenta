import os
import shutil
import YoloModel as YM
from utils_detection import *
#safe use only with THETIS dataset

PATH_TO_TRAINING = "C:\Licenta\VIDEO_RGB"
PATH_TO_SAVE = "C:\Licenta\Dataset"

model = YM.YoloModel()
model.training_mode = True

def get_video_name(path):
    path_splitted = path.split('\\')
    video_name_with_extension = path_splitted[len(path_splitted) - 1]
    video_name = video_name_with_extension.split(".")[0]
    return video_name

def make_csv_from_folders():

    all_folders_input = os.listdir("C:\Licenta\VIDEO_RGB")

    for folder_name in all_folders_input[4:5]:
        print(f"I process {folder_name}")
        path_output = f"{PATH_TO_SAVE}\\{folder_name}\\"
        path_current_folder = f"{PATH_TO_TRAINING}\\{folder_name}\\"

        path_saving_folder = f"{PATH_TO_SAVE}\\{folder_name}\\"

        if os.path.exists(path_output):
            if len(os.listdir(path_output)):
                print("The folder is not empty")
                return
            shutil.rmtree(path_output)

        os.mkdir(f"{PATH_TO_SAVE}\\{folder_name}\\")
        
        for file_name in os.listdir(path_current_folder):
           
            video_path_complete = f"{path_current_folder}\{file_name}"
            video_name = get_video_name(video_path_complete)
            
            print(f"Processing {video_path_complete}")

            all_detection, yolo_boxes = model.read_from_video(video_path_complete)

            save_to_csv_limbs(f"{path_output}\\{video_name}_limbs.csv", all_detection)
            save_to_csv_YOLO(f"{path_output}\\{video_name}_yolo.csv", yolo_boxes)

make_csv_from_folders()