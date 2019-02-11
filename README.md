# Manual Camera Calibration
This tool is implemented to find manually the camera calibration parameters, with a simple user interface that use keyboard keys to play with parameters and see the results online.

# How to run:
python3 src/manual_calib.py input_image

# How to use it:
The calibration is divided into 3 steps:

1. Undistort the input: The radial distortion parameters will be estimated by playing with k1/k2 params:

- I. Use Up/Down arrow keys to Increase/Decrease k1 parameter (first-order radial distortion) 
- II. Use Right/Left arrow keys to Increase/Decrease k2 parameter (second-order radial distortion) 
- III. Use +/- to change focal lenght, it will affect undistorting output
- IV. Click two points on image and press Space Key to draw a straight line, you can use this straight line as a ground truth.


2. Draw an square on real world coordinates:
The square coordinates should be clicked one by one, after each click press Enter to proceed;

- The 1st click sets the Top Left corner.
- The 2nd click sets the Top Right corner.
- The 3rd click sets the Bottom Left corner.
- The 4th click sets the Bottom Right corner.
Press Enter again to finish and see the top view image.

3. Crop the image: Try to find a good cropping for your transformation by:
I. Use arrow keys to displace the result image
II. Use +/- to zoom In/Out 

The results will be written in a text file:
./result.txt
