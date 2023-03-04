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

    plot_one_box(yolo_boxes, image_to_draw, label=label)

    return image_to_draw

def video_write(path_to_video_original, yolo_boxes, labels, confidence, sequence_lenght = 16):
    width, height, fps = None, None, None
    cap = cv2.VideoCapture(path_to_video_original)

    if cap.isOpened() == False:
        print("Error in opening file")
        return
    else:
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))

    vid_write_image = letterbox(cap.read()[1], 960, stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]

    capWriter = cv2.VideoWriter(f"keypoint3_model_shallow.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (resize_width, resize_height))
    number_frame = -1

    while cap.isOpened():
        ret, frame = cap.read()
        number_frame+=1
        if ret == True:
            image = letterbox(frame, 960, stride=64, auto=True)[0]
            image_copy = plot_yolo_boxes(image, yolo_boxes[number_frame], f"{labels[number_frame//sequence_lenght]}: {confidence[number_frame//sequence_lenght]}")
            capWriter.write(image_copy)
        else:
            break

    cap.release()
    capWriter.release()

def decode_output(number_label):
        if number_label == 0:
            return "backhand", "backhand"
        if number_label == 1:
            return 'backhand_slice', "backhand"
        if number_label == 2:
            return "backhand_volley", "backhand"
        if number_label == 3:
            return 'backhand2hands', "backhand"
        if number_label == 4:
            return 'flat_service', "service"
        if number_label == 5:
            return 'forehand_flat', "forehand"
        if number_label == 6:
            return 'forehand_openstands', "forehand"
        if number_label == 7:
            return 'forehand_slice', "forehand"
        if number_label == 8:
            return 'forehand_volley', "forehand"
        if number_label == 9:
            return 'kick_service', "service"
        if number_label == 10:
            return 'slice_service', "service"
        if number_label == 11:
            return 'smash'
        if number_label == -1:
            return "no_move"
        return None