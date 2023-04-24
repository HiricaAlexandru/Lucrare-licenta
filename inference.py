import sys

sys.path.append("F:\Licenta\Lucrare-licenta\yolov7\\utils\\")
sys.path.append("F:\Licenta\Lucrare-licenta\yolov7\\")

import models_LSTM as models
import torch
import YoloModel as YM
import utils_detection
from torch.utils.data import DataLoader
from torch import nn
from Vizualize import *
import pyodbc
import matplotlib.pyplot as plt
from zipfile import *
import os
import pandas as pd
import hashlib
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from threading import Thread

def make_video_labels():
    classes_name = dict()
    for i in range(12):
        classes_name[decode_output(i)[0]] = 0
    classes_name['no_move'] = 0
    return classes_name

def clear_video_labels():
    for elem in CLASSES_NAME.keys():
        CLASSES_NAME[elem] = 0


cnxn, cursor = None, None
model, model_YOLO = None, None
CLASSES_NAME = None
SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE = None, None, None
device = None
STATUS_LABEL, SELECT_BUTTON = None, None

VIDEO_PATH = "C:\\Users\\AlexH\\Downloads\\tennis_match_crop.mp4"
#VIDEO_PATH = "F:\Licenta\\test_videos\halep_cut.mp4"
#VIDEO_PATH = "F:\Licenta\\test_videos\Djokovic.mp4"
#VIDEO_PATH = "C:\\Users\\AlexH\\Downloads\\bojana.mp4"
#VIDEO_PATH = "C:\\Users\\AlexH\\Downloads\\Federer1.mp4"
NAME_OF_OUTPUT = ".tmp/video_labeled"
REVERSED = False
#VIDEO_PATH = "F:\Licenta\VIDEO_RGB\\backhand_slice\\p20_bslice_s2.avi"
#VIDEO_PATH = "F:\\Licenta\\VIDEO_RGB\\backhand_volley\\p1_bvolley_s2.avi"
#VIDEO_PATH = "F:\Licenta\Lucrare-licenta\p20_bslice_s2.avi"

def create_folders():
    if not os.path.isdir("saved_videos"):
        os.mkdir("saved_videos")

    if not os.path.isdir(".tmp"):
        os.mkdir(".tmp")

def create_database_connection():
    cnxn_str = ("Driver={SQL Server};"
                "Server=DESKTOP-7IPHS13\SQLEXPRESS;"
                "Database=VideoEntry;"
                "Trusted_Connection=yes;")
    cnxn = pyodbc.connect(cnxn_str)
    cursor = cnxn.cursor()

    return cnxn, cursor

def make_predictions_video(video_original_path, path_video_to_save):
    global STATUS_LABEL

    STATUS_LABEL['text'] = f"Finding the person in the video {video_original_path}"
    all_detection, yolo_boxes = model_YOLO.read_from_video(video_original_path)
    print("Read video file")

    all_detections_normalized = utils_detection.normalize_detection_limbs(yolo_boxes, all_detection)

    all_detections_normalized = utils_detection.convert_to_2D_matrix(all_detections_normalized)
    all_detections_sequence, _ = utils_detection.get_all_sequences_from_2D_format(all_detections_normalized, SEQUENCE_LENGTH, 0, SEQUENCE_LENGTH)
    all_detections_sequence = torch.tensor(all_detections_sequence).float().to(device)

    output_labels = []
    confidence = []

    STATUS_LABEL['text'] = f"Predicting the action from {video_original_path}"
    test_loader = DataLoader(all_detections_sequence, 1, shuffle=False)
    number = 0
    with torch.no_grad():
        for X in test_loader:
            number+=1
            output = model(X)
            softmax = nn.Softmax(dim = 1)
            output = softmax(output)
            maximum_values, predicted = torch.max(output.data, 1)
            for i in range(len(predicted)):
                if maximum_values[i] < 0.7:
                    predicted[i] = -1
            
            output_labels.append(predicted[i].item())
            confidence.append(maximum_values[i].item())


    output_names = [None for i in range(len(output_labels))]

    for i in range(len(output_labels)):
        if REVERSED == True:
            output_names[i] = decode_output(output_labels[i])[2]
        else:
            output_names[i] = decode_output(output_labels[i])[0]

    STATUS_LABEL['text'] = "Writing resulting video file"
    video_write(video_original_path, yolo_boxes, output_names, confidence, path_video_to_save, SEQUENCE_LENGTH)
    
    return output_names

def search_hash_in_database(hash):
    global cursor, cnxn
    cursor.execute(f"SELECT * FROM EntryVideo WHERE HASH = '{hash}';")

    rows = cursor.fetchall()
    
    if len(rows) > 0:
        for row in rows:
            #print(row, type(row), row.)
            row_to_list = [elem.strip() for elem in row]
            return row_to_list[1]
    else:
        return False

def delete_entry_database(primary_key):
    global cursor, cnxn
    cursor.execute(f"DELETE FROM EntryVideo WHERE HASH ='{primary_key}';")
    cnxn.commit()

def insert_value(hash, video_path):
    global cursor, cnxn
    cursor.execute(f"INSERT INTO EntryVideo VALUES('{hash}', '{video_path}');")
    cnxn.commit()

def fill_outputs_value(outputs):
    for value in outputs:
        CLASSES_NAME[value] += 1

def make_csv_and_graph(name_of_output):
    labels = list(CLASSES_NAME.keys())
    values = list(CLASSES_NAME.values())

    df = pd.DataFrame(CLASSES_NAME.items())
    df.to_csv('.tmp/Shot_occurance.csv')

    fig, ax = plt.subplots(figsize = (20, 10))
    ax.bar(labels, values)
    #plt.tight_layout()
    plt.xlabel("Name of hit")
    plt.ylabel("Occurance")
    plt.xticks(rotation=45)
    plt.title("Number of hits per class")
    plt.savefig(".tmp/Shot_occurance")

    with ZipFile(f'saved_videos/{name_of_output}.zip', 'w') as myzip:
        myzip.write('.tmp/Shot_occurance.png', 'Shot_occurance.png')
        myzip.write('.tmp/Shot_occurance.csv', 'Shot_occurance.csv')
        myzip.write(".tmp/video_labeled.mp4", 'video_labeled.mp4')
    
    os.remove(".tmp/Shot_occurance.png")
    os.remove(".tmp/Shot_occurance.csv")
    os.remove(".tmp/video_labeled.mp4")

def get_hash_of_file(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def init():
    global cnxn, cursor, model, model_YOLO, device, SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE, CLASSES_NAME

    MODEL_PATH = "F:\\Licenta\\Lucrare-licenta\\results\saved_checkpoint_LSTM_60_epoch.pth"
    create_folders()
    cnxn, cursor = create_database_connection()
    CLASSES_NAME = make_video_labels()

    SEQUENCE_LENGTH, INPUT_SIZE, HIDDEN_SIZE = models.LSTM.return_train_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.LSTM(INPUT_SIZE, hidden_units=HIDDEN_SIZE, seq_length=SEQUENCE_LENGTH).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print("Loaded the LSTM model")

    model_YOLO = YM.YoloModel()
    model_YOLO.inference = True
    model_YOLO.training_mode = True

    print("Loaded the YOLO model")

def create_output_name_of_video(video_path):
    time_now = datetime.now()
    date_time = time_now.strftime("%m-%d-%Y")
    name = utils_detection.get_video_name(video_path)

    return f"{name}-{date_time}"

def make_predictions(video_path):
    global STATUS_LABEL
    hash_of_file = get_hash_of_file(video_path)
    hash_file = search_hash_in_database(hash_of_file)
    name_of_output = NAME_OF_OUTPUT
    print(video_path)
    if hash_file != False :
        if not(os.path.exists(hash_file)):
            #if the file path is no longer existing we need to delete from the database this entry
            delete_entry_database(hash_of_file)
        else:
            print(f"It is found at {hash_file}")
            STATUS_LABEL['text'] = f"The file if found at {hash_file}"
            SELECT_BUTTON["state"] = "normal"
            return False


    outputs = make_predictions_video(video_path, name_of_output)
    fill_outputs_value(outputs)
    name_of_output_video = create_output_name_of_video(video_path)

    STATUS_LABEL['text'] = "Creating the zip file"
    make_csv_and_graph(name_of_output_video)
    path_to_video = os.path.abspath(f"saved_videos/{name_of_output_video}.zip")
    insert_value(hash_of_file, path_to_video)
    STATUS_LABEL['text'] = f"Finished, zip file at location {path_to_video}"
    SELECT_BUTTON["state"] = "normal"

    return True

def browse_file_run_inference():
    file_types = [
        ("Avi files", "*.avi"),
        ("Mp4 files", "*.mp4")
    ]

    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = file_types)
    print(SELECT_BUTTON)
    SELECT_BUTTON["state"] = "disabled"
    filename = filename.replace('/', '\\')
    thread = Thread(target=make_predictions, args=[filename])
    thread.start()
    #make_predictions(filename)


def GUI_run():
    global SELECT_BUTTON, STATUS_LABEL
    
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    root = tk.Tk()
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")  # Set the size of the window
    # Create a label widget to display the message

    frm = tk.Frame(root, padx=10, pady=10)
    
    frm.grid()
    tk.Label(frm, text="Select file to analyze", pady = 10).grid(column=0, row=0)
    SELECT_BUTTON = tk.Button(frm, text="Select file", command=browse_file_run_inference, pady = 10)
    SELECT_BUTTON.grid(column=0, row=1)
    STATUS_LABEL = tk.Label(frm, text="", pady = 10)
    STATUS_LABEL.grid(column=0, row=2)
    frm.place(relx=0.5, rely=0.5, anchor="center")
    root.mainloop()

init()
GUI_run()
#make_predictions(VIDEO_PATH)


cursor.close()
cnxn.close()
