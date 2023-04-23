from plots import plot_one_box, plot_skeleton_kpts
import cv2
from datasets import letterbox

def visualize_box_detection(detection, algo_output, image, label = "Default"):

    if len(algo_output.shape) == 1:
        algo_output = algo_output.reshape(1, algo_output.shape[0])

    if len(detection.shape) == 1:
        detection = detection.reshape(1, detection.shape[0])

    image_to_draw = image.copy()

    for idx in range(detection.shape[0]):
        plot_one_box(detection[idx], image_to_draw, label=label)
    
    for idx in range(detection.shape[0]):
        plot_skeleton_kpts(image_to_draw, algo_output[idx, 7:].T, 3)

    return image_to_draw

def plot_yolo_boxes(image, yolo_boxes, label = "Default"):
    image_to_draw = image.copy()
    
    if yolo_boxes[0] == -1 and yolo_boxes[1] == -1 and yolo_boxes[2] == -1 and yolo_boxes[3] == -1:
        return image_to_draw
    
    plot_one_box(yolo_boxes, image_to_draw, label=label, color=[255,0,0]) #to be red

    return image_to_draw

def video_write(path_to_video_original, yolo_boxes, labels, confidence, name_of_output, sequence_lenght = 16):
    width, height, fps = None, None, None
    cap = cv2.VideoCapture(path_to_video_original)

    if cap.isOpened() == False:
        print("Error in opening file")
        return
    else:
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))

    ret, frame = cap.read()
    vid_write_image = letterbox(frame, 960, stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]

    capWriter = cv2.VideoWriter(f"{name_of_output}.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (resize_width, resize_height))
    number_frame = 0
    frame = vid_write_image
    
    length_of_sequences = len(labels)

    while cap.isOpened():
        if ret == True:
            image = letterbox(frame, 960, stride=64, auto=True)[0]
            image_copy = image
            if number_frame//sequence_lenght < length_of_sequences:
                image_copy = plot_yolo_boxes(image, yolo_boxes[number_frame], f"{labels[number_frame//sequence_lenght]}")
            else:
                try:
                    image_copy = plot_yolo_boxes(image, yolo_boxes[number_frame], "NoValues")
                except:
                    pass
                        
            capWriter.write(image_copy)
        else:
            break

        ret, frame = cap.read()
        number_frame+=1


    print(number_frame)
    cap.release()
    capWriter.release()

def decode_output(number_label):
        if number_label == 0:
            return "backhand", "backhand", 'forehand_flat', "forehand"
        if number_label == 1:
            return 'backhand_slice', "backhand", 'forehand_slice', "forehand"
        if number_label == 2:
            return "backhand_volley", "backhand", 'forehand_volley', "forehand"
        if number_label == 3:
            return 'backhand2hands', "backhand", 'forehand2hands', 'forehand'
        if number_label == 4:
            return 'flat_service', "service", 'flat_service', "service"
        if number_label == 5:
            return 'forehand_flat', "forehand", "backhand", "backhand"
        if number_label == 6:
            return 'forehand_openstands', "forehand", 'backhand2hands', "backhand"
        if number_label == 7:
            return 'forehand_slice', "forehand", 'backhand_slice', "backhand"
        if number_label == 8:
            return 'forehand_volley', "forehand", "backhand_volley", "backhand"
        if number_label == 9:
            return 'kick_service', "service", 'kick_service', "service"
        if number_label == 10:
            return 'slice_service', "service", 'slice_service', "service"
        if number_label == 11:
            return 'smash', 'smash', 'smash', 'smash'
        if number_label == -1:
            return "no_move", 'no_move', "no_move", 'no_move'
        return None