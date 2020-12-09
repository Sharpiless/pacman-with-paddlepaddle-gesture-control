import cv2
import imutils
import numpy as np
import argparse
bg = None


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype('float')
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main(dtype):
    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    top, right, bottom, left = 90, 380, 285, 590

    num_frames = 0
    thresholded = None

    count = 0

    while(True):
        (grabbed, frame) = camera.read()
        if grabbed:

            frame = imutils.resize(frame, width=700)

            frame = cv2.flip(frame, 1)

            clone = frame.copy()

            (height, width) = frame.shape[:2]

            roi = frame[top:bottom, right:left]

            thresholded = roi

            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            num_frames += 1

            cv2.imshow('Video Feed', clone)
            if not thresholded is None:
                cv2.imshow('Thesholded', thresholded)

            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord('q'):
                break

            if keypress == ord('s'):
                cv2.imwrite('data/{}/saved_v1_{:04}.jpg'.format(dtype, count), thresholded)
                count += 1
                print(count, 'saved.')

        else:
            camera.release()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='down')
    args = parser.parse_args()
    main(args.dtype)
    cv2.destroyAllWindows()
