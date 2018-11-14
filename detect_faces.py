import sys, getopt
from Detector import Detector
import cv2


def main(argv):
    vide_file = ''
    try:
        opts, args = getopt.getopt(argv, "hi", ["video_file="])
    except getopt.GetoptError:
        print('detect_faces.py -i <video_file>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('detect_faces.py -i <video_file>')
            sys.exit()
        elif opt in ('-i', '--video_file'):
            vide_file = arg

    # Firing up the detector
    detector = Detector(graph='models/frozen_graph.pb', labels='models/face_label_map.pbtxt')

    # Initialize video stream
    stream = cv2.VideoCapture(vide_file)
    # Get number of frames from the video
    number_of_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Loop and process every frame
    for i in range(number_of_frames):

        ret, frame = stream.read()
        if ret:
            frame, scores, num_detections, boxes = detector.detect(frame, resizing_factor=4)
            cv2.imshow('frame', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__=='__main__':

    main(sys.argv[1:])