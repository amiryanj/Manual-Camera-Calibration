import os, sys
import cv2
import numpy as np

if len(sys.argv) < 3:
    print('Please enter URL for the input image')
# Load image
input_file = sys.argv[1]
params_file = sys.argv[2]
# input_file = '/home/jamirian/workspace/pamela/PAMELA_video_samples/P3365 CAM4 17_06_2019 11_24_36 1.mp4'

print(input_file)
print(params_file)

assert(os.path.isfile(input_file))
filename, file_extension = os.path.splitext(input_file)


param_content = np.load(params_file)
H, camMatrix, distCoeffs = param_content['H'], param_content['camMatrix'], param_content['distCoeffs']

print(distCoeffs)


def warp(src, H, cam_matrix, dist_coeffs):
    ud = cv2.undistort(frame, cam_matrix, dist_coeffs)
    out = cv2.warpPerspective(ud, H, (src.shape[1], src.shape[0]),
                              cv2.INTER_LINEAR)
    return out


if file_extension.lower() in ['.png', '.jpg', '.jpeg']:
    pass

elif file_extension.lower() in ['.mp4', '.avi']:
    vid_cap = cv2.VideoCapture(input_file)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_wrt = cv2.VideoWriter(filename + '_warped' + file_extension, fourcc, fps, (width,height))

    while True:
        ret, frame = vid_cap.read()
        if not ret: break
        warped = warp(frame, H, camMatrix, distCoeffs)
        vid_wrt.write(warped)
        cv2.imshow('undistort', warped)
        key = cv2.waitKey(10)
        if key == 'q':
            break

    vid_wrt.release()
