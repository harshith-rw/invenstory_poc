import cv2
import matplotlib.pyplot as plt
import pytesseract

from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to the Tesseract executable (replace with your path)
pytesseract.pytesseract.tesseract_cmd = r'C:\using_opencv\tesseract'

img = cv2.imread("image.jpg", cv2.IMREAD_COLOR)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
result_image = img.copy()
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# Assume the largest contour represents the medicine box
largest_contour = max(contours, key=cv2.contourArea)

# Extract the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the region of interest (ROI)
medicine_box = img[y:y+h, x:x+w]

# Display the cropped medicine box
plt.imshow(cv2.cvtColor(medicine_box, cv2.COLOR_BGR2RGB))
plt.title('Medicine Box')
plt.show()

# Use OCR to extract text from the cropped image
text = pytesseract.image_to_string(medicine_box)

# Print the extracted text
print("Extracted Text:", text)



# # Convert the image to bytes
# image_bytes = cv2.imencode('image.jpg', img)[1].tobytes()

# # Initialize the Google Cloud Vision client
# client = vision.ImageAnnotatorClient()

# # Create a Google Cloud Vision image object
# image_google = types.Image(content=image_bytes)

# # Perform OCR on the image
# response = client.text_detection(image=image_google)
# texts = response.text_annotations

# # Print the detected text
# for text in texts:
#     print(text.description)

# # Optionally, you can visualize the detected text on the image
# for text in texts:
#     vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
#     cv2.polylines(img, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)

# # Display the image with detected text using OpenCV
# cv2.imshow('Detected Text', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

