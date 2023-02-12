from plots import plot_one_box
import cv2

def visualize_box_detection(detection, image):

    for idx in range(detection.shape[0]):
        plot_one_box(detection[idx], image)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()
        