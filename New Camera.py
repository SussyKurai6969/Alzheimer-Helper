import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

def action_detection(event, x, y, flags, param):
    # Here, you can define specific actions for the detected motion
    pass

# Load the image
img = cv2.imread('path/to/image.jpg')

# Create a window and bind the functions to it
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the area of interest
x, y, w, h = 200, 200, 400, 400

# Define the number of frames to use for background subtraction
n_frames = 60

# Initialize the background
for i in range(n_frames):
    bg = cv2.flip(gray, 1) if i % 2 == 0 else gray
    bg = bg[y:y+h, x:x+w]

# Initialize the variable to keep track of the moving object
moving_object = np.zeros((h, w), np.uint8)

# Detect the motion
while True:
    # Get the current frame
    frame = cv2.flip(gray, 1) if i % 2 == 0 else gray
    frame = frame[y:y+h, x:x+w]

    # Apply the background subtraction
    fgmask = cv2.absdiff(bg, frame)
    fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]

    # Apply a blur to the moving object
    fgmask = cv2.medianBlur(fgmask, 15)

    # Draw circles around the detected moving object
    circles = cv2.HoughCircles(fgmask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=400)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0]+x, i[1]+y), i[2], (0, 255, 0), 2)
            cv2.putText(img, "Detected Moving Object", (i[0]+x, i[1]+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original image with the moving object detected
    cv2.imshow('image', img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('C:\Vaibhavs Codes\Python\School Project\Security System\New Recording', fourcc, 20.0, (img.shape[1], img.shape[0]))

# Your existing code for the loop
while True:
    # Your previous code for processing the frames

    # Write the processed frame to the video file
    out.write(img)

    # Display the processed frame
    cv2.imshow('image', img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter object and close the output video file
out.release()

cv2.destroyAllWindows()