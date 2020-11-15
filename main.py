import cv2
import numpy as np

# source: https://stackoverflow.com/a/44659589
def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def overlay_watermark(frame, watermark):
    frame_h, frame_w, frame_c = frame.shape
    overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')

    watermark_h, watermark_w, watermark_c = watermark.shape

    offset_w = frame_w - watermark_w - 20
    offset_h = frame_h - watermark_h - 20

    for i in range(0, watermark_h):
        for j in range(0, watermark_w):
            if watermark[i,j][3] != 0:
                overlay[i+offset_h, j+offset_w] = watermark[i, j]

    cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, frame)
    return frame


def main():
    VIDEO_FILE = 'driving.mp4'
    WATERMARK_FILE = "watermark.png"

    watermark = cv2.imread(WATERMARK_FILE)
    watermark = resize_image(watermark, height=50)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
    cv2.imshow('water', watermark)


    capture = cv2.VideoCapture(VIDEO_FILE)
    capture.set(3, 1280)
    capture.set(4, 720)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.avi', fourcc, capture.get(cv2.CAP_PROP_FPS), 
        (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = capture.read()
        if ret is True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            frame = overlay_watermark(frame, watermark)
            cv2.imshow('frame', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            output.write(frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break

    capture.release()
    output.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()