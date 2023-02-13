from plots import plot_one_box, plot_skeleton_kpts
import cv2

def visualize_box_detection(detection, algo_output, image):

    if len(algo_output.shape) == 1:
        algo_output = algo_output.reshape(1, algo_output.shape[0])

    if len(detection.shape) == 1:
        detection = detection.reshape(1, detection.shape[0])

    image_to_draw = image.copy()

    for idx in range(detection.shape[0]):
        plot_one_box(detection[idx], image_to_draw, label="ALABALA")
    
    for idx in range(detection.shape[0]):
        plot_skeleton_kpts(image_to_draw, algo_output[idx, 7:].T, 3)

    return image_to_draw
        