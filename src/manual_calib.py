import cv2
import numpy as np
from src.DEFINITIONS import *


# Load image
im = cv2.imread('/home/jamirian/Pictures/KTG_K2_resized.png')
# cv2.imshow('im', im)

pnt_TL = [-50, 0]
pnt_TR = [-50, 0]
pnt_BL = [-50, 0]
pnt_BR = [-50, 0]
zz = 0
grid_pts = []
for yy in range(-10, 10):
    grid_pts.append([-20, yy, zz])
    grid_pts.append([20, yy, zz])

for xx in range(-20, 20, 2):
    grid_pts.append([xx, -10, zz])
    grid_pts.append([xx, 10, zz])
grid_pts = np.float32(grid_pts)

window_titles = ["Click Top Left Point",
                 "Click Top Right Point",
                 "Click Bottom Left Point",
                 "Click Bottom Right Point"]
text = ['']

def drawLineSeg(src):
    im_copy = src.copy()
    # if line_seg_completed[0]:
    cv2.line(im_copy, (line_seg[0], line_seg[1]), (line_seg[2], line_seg[3]), BLUE_COLOR, 2)

    return im_copy


def drawRect(src):
    im_copy = src.copy()

    if pnt_TR[0] >= 0:
        cv2.line(im_copy, (pnt_TL[0], pnt_TL[1]), (pnt_TR[0], pnt_TR[1]), BLUE_COLOR, 2)
    if pnt_BL[0] >= 0:
        cv2.line(im_copy, (pnt_TL[0], pnt_TL[1]), (pnt_BL[0], pnt_BL[1]), BLUE_COLOR, 2)
    if pnt_BR[0] >= 0:
        cv2.line(im_copy, (pnt_TR[0], pnt_TR[1]), (pnt_BR[0], pnt_BR[1]), BLUE_COLOR, 2)
        cv2.line(im_copy, (pnt_BL[0], pnt_BL[1]), (pnt_BR[0], pnt_BR[1]), BLUE_COLOR, 2)

    cv2.circle(im_copy, (pnt_TL[0], pnt_TL[1]), 10, RED_COLOR, 2)
    cv2.circle(im_copy, (pnt_TR[0], pnt_TR[1]), 10, RED_COLOR, 2)
    cv2.circle(im_copy, (pnt_BL[0], pnt_BL[1]), 10, RED_COLOR, 2)
    cv2.circle(im_copy, (pnt_BR[0], pnt_BR[1]), 10, RED_COLOR, 2)
    return im_copy


def click_rectangle(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        if text[0] == window_titles[0]:
            pnt_TL[0], pnt_TL[1] = x, y
        elif text[0] == window_titles[1]:
            pnt_TR[0], pnt_TR[1] = x, y
        elif text[0] == window_titles[2]:
            pnt_BL[0], pnt_BL[1] = x, y
        elif text[0] == window_titles[3]:
            pnt_BR[0], pnt_BR[1] = x, y

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        pass


line_seg = [0,-1, 0, -1]
line_seg_completed = [True]
def click_line(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        if line_seg_completed[0]:
            line_seg[0] = x
            line_seg[1] = y
            line_seg_completed[0] = False
        else:
            line_seg[2] = x
            line_seg[3] = y
            line_seg_completed[0] = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        pass


def calibrate_with_rect(src):
    for ii in range(4):
        while True:
            text[0] = window_titles[ii]
            cv2.namedWindow(window_titles[ii])
            cv2.moveWindow(window_titles[ii], 20, 20)
            cv2.setMouseCallback(window_titles[ii], click_rectangle)

            im_copy = drawRect(src)
            cv2.imshow(window_titles[ii], im_copy)
            key = cv2.waitKeyEx()
            if key == ENTER_KEY:
                cv2.destroyAllWindows()
                break

    im_copy = drawRect(src)

    SCALE = 720
    srcPts = np.float32([[0,0], [SCALE,0], [0,SCALE], [SCALE,SCALE]]) + np.array([200, 200])
    dstPts = np.float32([[pnt_TL[0], pnt_TL[1]], [pnt_TR[0], pnt_TR[1]],
                         [pnt_BL[0], pnt_BL[1]], [pnt_BR[0], pnt_BR[1]]])
    H, mask = cv2.findHomography(srcPts, dstPts)

    prj_grid = cv2.perspectiveTransform(np.reshape(grid_pts, (-1, 1, 3))[:, :, :2], H)
    prj_grid = np.float32(prj_grid).reshape(-1, 2)

    for ii in range(0, len(prj_grid), 2):
        ptA = (prj_grid[ii][0], prj_grid[ii][1])
        ptB = (prj_grid[ii + 1][0], prj_grid[ii + 1][1])
        if ptA[0] > 10000 or ptA[0] < -10000:
            continue
        cv2.circle(im_copy, ptA, 3, (200, 100, 0), 2)
        cv2.circle(im_copy, ptB, 3, (0, 100, 200), 2)
        cv2.line(im_copy, ptA, ptB, (100, 200, 0), 1)

    cv2.imshow('Floor Plane', im_copy)
    cv2.moveWindow('Floor Plane', 20, 20)
    cv2.waitKeyEx()
    cv2.destroyAllWindows()

    return H


# Manual Calibration
def calcRMat(theta_x, theta_y, theta_z):
    theta_x = theta_x * PI / 180.
    theta_y = theta_y * PI / 180.
    theta_z = theta_z * PI / 180.

    Rx = np.eye(3,3)
    Rx[1, 1] = np.cos(theta_x)
    Rx[1, 2] = np.sin(theta_x)
    Rx[2, 1] = -np.sin(theta_x)
    Rx[2, 2] = np.cos(theta_x)
    # print('Rx = \n', Rx)

    Ry = np.eye(3,3)
    Ry[0, 0] = np.cos(theta_y)
    Ry[0, 2] = -np.sin(theta_y)
    Ry[2, 0] = np.sin(theta_y)
    Ry[2, 2] = np.cos(theta_y)
    # print('Ry = \n', Ry)

    Rz = np.eye(3,3)
    Rz[0, 0] = np.cos(theta_z)
    Rz[0, 1] = np.sin(theta_z)
    Rz[1, 0] = -np.sin(theta_z)
    Rz[1, 1] = np.cos(theta_z)
    # print('Rz = \n', Rz)

    return np.matmul(Rz, np.matmul(Ry, Rx))


def calcCamMatrix(f, im_size):
    K = np.eye(3)
    K[0, 0] = f
    K[0, 2] = im_size[1] / 2
    K[1, 1] = f
    K[1, 2] = im_size[0] / 2

    return K


def project_points(pts, RT, K):
    prj_pts = []
    for pt in pts:
        rt_pt = np.matmul(RT, np.hstack((pt, 1)))
        k_pt = np.matmul(K, rt_pt)
        prj_pts.append(np.array([k_pt[0]/k_pt[2], k_pt[1]/k_pt[2]]).astype(int))

    return prj_pts

theta_deg = np.array([[180], [0], [0]], dtype=float)
RMat = calcRMat(theta_deg[0], theta_deg[1], theta_deg[2])
CamLoc = np.array([[0], [0], [20]], dtype=float)
Rt = np.hstack((RMat, CamLoc))
f = 1500
k1 = 0
k2 = 0


def undistort(src, f, k1k2):
    K = calcCamMatrix(f, src.shape)
    im_undistort = cv2.undistort(src, K, k1k2)
    return im_undistort

while True:
    out = undistort(im, f, np.array([k1, k2, 0, 0, 0]))
    out_copy = drawLineSeg(out)
    cv2.imshow('undistort', out_copy)
    cv2.namedWindow('undistort')
    cv2.moveWindow('undistort', 20, 20)
    cv2.setMouseCallback('undistort', click_line)
    key = cv2.waitKeyEx()
    if key == PLUS_Key:
        f *= 1.05
    elif key == MINUS_Key:
        f /= 1.05
    elif key == UP_ARROW_KEY:
        k1 -= 0.05
    elif key == DOWN_ARROW_KEY:
        k1 += 0.05
    elif key == RIGHT_ARROW_KEY:
        k2 += 0.05
    elif key == LEFT_ARROW_KEY:
        k2 -= 0.05
    elif key == ENTER_KEY:
        im = out
        print('distortion params [k1, k2] = [%f, %f]' %(k1, k2))
        break

    elif key == SPACE_KEY:
        cv2.destroyAllWindows()

    elif key == ESCAPE_KEY:
        break


# Call Calibration function
H_Mat = calibrate_with_rect(im)
print('Homography Estimated = \n', H_Mat, '\n************************\n')

# Crop Image
# affineMat = np.zeros((2, 3), dtype=float)
# affineMat[0,0], affineMat[1,1] = 1, 1
affineMat = np.eye(3, dtype=float)
while True:
    homog_and_affine = np.matmul(H_Mat, affineMat)
    H2_out = cv2.warpPerspective(im, homog_and_affine, (im.shape[1], im.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    cv2.imshow('warped', H2_out)
    key = cv2.waitKeyEx()

    if key == MINUS_Key:
        affineMat[0, 0] *= 1.05
        affineMat[1, 1] *= 1.05
    elif key == PLUS_Key:
        affineMat[0, 0] /= 1.05
        affineMat[1, 1] /= 1.05
    if key == RIGHT_ARROW_KEY:
        affineMat[0, 2] -= 5
    if key == LEFT_ARROW_KEY:
        affineMat[0, 2] += 5
    if key == UP_ARROW_KEY:
        affineMat[1, 2] += 5
    if key == DOWN_ARROW_KEY:
        affineMat[1, 2] -= 5

    elif key == ESCAPE_KEY:
        break
    elif key == ENTER_KEY:
        H_Mat = homog_and_affine
        break

print('Perspective Transformation After Cropping = \n', H_Mat, '\n************************')

exit(1)

while True:

    K = calcCamMatrix(f, im.shape)
    RMat = calcRMat(theta_deg[0], theta_deg[1], theta_deg[2])
    Tr = np.matmul(RMat.transpose(), CamLoc) * -1.
    Rt = np.hstack((RMat.transpose(), Tr))

    im_copy = im.copy()
    proj_pts = project_points(grid_pts, Rt, K)
    for i in range(0, len(proj_pts), 2):
        ptA = (proj_pts[i][0], proj_pts[i][1])
        ptB = (proj_pts[i + 1][0], proj_pts[i + 1][1])
        if ptA[0] > 10000 or ptA[0] < -10000:
            continue
        cv2.circle(im_copy, ptA, 3, (200, 100, 0), 2)
        cv2.circle(im_copy, ptB, 3, (0, 100, 200), 2)
        cv2.line(im_copy, ptA, ptB, (100, 200, 0), 1)
    world_center = project_points([[0, 0, 0]], Rt, K)
    if -2000 < world_center[0][0] < 2000 and -2000 < world_center[0][1] < 2000:
        cv2.circle(im_copy, (world_center[0][0], world_center[0][1]), 10, (50, 0, 240), 3)

    cv2.putText(im_copy, "Loc =" + np.array2string(CamLoc, floatmode='unique'),
                (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.putText(im_copy, "Ang" + np.array2string(theta_deg, floatmode='unique'),
                (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imshow('im', im_copy)
    key = cv2.waitKeyEx(0)

    # if key == RIGHT_ARROW_KEY:
    #     theta_deg[1] += 1
    # elif key == LEFT_ARROW_KEY:
    #     theta_deg[1] -= 1
    # elif key == UP_ARROW_KEY:
    #     theta_deg[0] += 1
    # elif key == DOWN_ARROW_KEY:
    #     theta_deg[0] -= 1

    dist = 30.
    if key == RIGHT_ARROW_KEY:
        theta_deg[2] += 1
    elif key == LEFT_ARROW_KEY:
        theta_deg[2] -= 1
    elif key == UP_ARROW_KEY:
        theta_deg[0] += 1  #  change Rx
    elif key == DOWN_ARROW_KEY:
        theta_deg[0] -= 1  #  change Rx

    CamLoc[0] = np.cos(theta_deg[0] * PI / 180.) * np.sin(theta_deg[2] * PI / 180.) * dist
    CamLoc[1] = np.sin(theta_deg[0] * PI / 180.) * np.sin(theta_deg[2] * PI / 180.) * dist
    CamLoc[2] = np.cos(theta_deg[2] * PI / 180.) * dist
    print(CamLoc)

    if key == 97:  # A
        CamLoc[1] -= 1
    elif key == 100:  # D
        CamLoc[1] += 1
    elif key == 115:  # S
        CamLoc[2] += 1
    elif key == 119:  # W
        CamLoc[2] -= 1

    elif key == PLUS_Key:
        f *= 1.05
    elif key == MINUS_Key:
        f /= 1.05

    elif key == ESCAPE_KEY:
        exit(1)


    print(key)
