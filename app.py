from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2

app = Flask(__name__, static_url_path='/static')

def image_normalization(image, percentile):
    image_normalized = np.zeros(image.shape, dtype='float64')
    for i in range(3):
        min_percentile = np.percentile(image[:,:,i], 100 - percentile)
        max_percentile = np.percentile(image[:,:,i], percentile)
        image_normalized[:,:,i] = (image[:,:,i] - min_percentile) / (max_percentile - min_percentile)
        image_normalized[:,:,i] = image_normalized[:,:,i] * 255   # Hardcoding this 255 as we want an 8-bit image.
    return image_normalized.astype('uint8')

def identify_elements_and_body(image):
    """
    Identify elements and the body in the image.
    Returns bounding boxes for elements and the body.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to identify elements and body
    min_area_threshold = 1000  # Adjust threshold based on image size and element size
    max_area_threshold = 50000  # Adjust threshold based on image size and element size
    element_contours = []
    body_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_threshold < area < max_area_threshold:
            element_contours.append(contour)
        elif area >= max_area_threshold:
            body_contour = contour

    return element_contours, body_contour

def extract_color_and_size_information(image, element_contours, body_contour):
    """
    Extract color and size information for elements and the body.
    Returns colors and sizes of elements and the body.
    """
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Initialize body color with zeros
    body_color = (0, 0, 0)

    # Extract color information for body if a contour is found
    if body_contour is not None:
        body_mask = np.zeros_like(image)
        cv2.drawContours(body_mask, [body_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        body_hsv = cv2.bitwise_and(hsv_image, body_mask)
        body_color = np.mean(body_hsv[:, :, 0]), np.mean(body_hsv[:, :, 1]), np.mean(body_hsv[:, :, 2])

    # Extract color and size information for elements
    element_colors = []
    element_sizes = []
    for contour in element_contours:
        # Extract color information
        element_mask = np.zeros_like(image)
        cv2.drawContours(element_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        element_hsv = cv2.bitwise_and(hsv_image, element_mask)
        element_color = np.mean(element_hsv[:, :, 0]), np.mean(element_hsv[:, :, 1]), np.mean(element_hsv[:, :, 2])
        element_colors.append(element_color)

        # Extract size information
        element_area = cv2.contourArea(contour)
        element_sizes.append(element_area)

    return element_colors, element_sizes, body_color

def calculate_balance_score(element_sizes, body_size):
    """
    Calculate the balance score based on the relative proportions of element sizes.
    Returns the balance score.
    """
    # Calculate relative proportions
    relative_proportions = [size / body_size for size in element_sizes]

    # Calculate balance score
    balance_score = 1 - np.std(relative_proportions)

    return balance_score

def calculate_proportion_score(element_sizes, body_size):
    """
    Calculate the proportion score based on the relative sizes of elements compared to the body.
    Returns the proportion score.
    """
    # Calculate relative sizes
    relative_sizes = [size / body_size for size in element_sizes]

    # Calculate proportion score
    proportion_score = np.mean(relative_sizes)

    return proportion_score

def calculate_symmetry_score(image):
    """
    Calculate the symmetry score of the image.
    Returns the symmetry score.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the horizontal mirror image
    mirror = cv2.flip(gray, 1)

    # Compute the absolute difference between the original image and its mirror
    diff = cv2.absdiff(gray, mirror)

    # Calculate the symmetry score
    symmetry_score = np.mean(diff)

    return symmetry_score

def calculate_simplicity_score(element_contours):
    """
    Calculate the simplicity score based on the number of elements.
    Returns the simplicity score.
    """
    # Calculate the number of elements
    num_elements = len(element_contours)

    # Calculate the simplicity score
    simplicity_score = 1 / (1 + num_elements)

    return simplicity_score

def calculate_harmony_score(element_colors):
    """
    Calculate the harmony score based on the color harmony of elements.
    Returns the harmony score.
    """
    # Calculate the variance of hue, saturation, and value channels
    hue_var = np.var([color[0] for color in element_colors])
    sat_var = np.var([color[1] for color in element_colors])
    val_var = np.var([color[2] for color in element_colors])

    # Calculate the harmony score
    harmony_score = 1 - (hue_var + sat_var + val_var) / (3 * 255 ** 2)

    return harmony_score

def calculate_contrast_score(element_colors, body_color):
    """
    Calculate the contrast score based on the color contrast between elements and the background.
    Returns the contrast score.
    """
    # Calculate the Euclidean distance between element colors and the body color
    distances = [np.linalg.norm(np.array(color) - np.array(body_color)) for color in element_colors]

    # Calculate the mean contrast
    mean_contrast = np.mean(distances)

    # Normalize the mean contrast
    normalized_contrast = mean_contrast / 255

    # Calculate the contrast score
    contrast_score = 1 - normalized_contrast

    return contrast_score

def calculate_unity_score(element_sizes):
    """
    Calculate the unity score based on the uniformity of element sizes.
    Returns the unity score.
    """
    # Calculate the standard deviation of element sizes
    size_std = np.std(element_sizes)

    # Calculate the unity score
    unity_score = 1 - size_std / np.mean(element_sizes)

    return unity_score

def process_image(image_data):
    """
    Process the input image to calculate aesthetic design indicators.
    """
    # Decode base64 image
    image_data = base64.b64decode(image_data.split(',')[1])

    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Identify elements and body
    element_contours, body_contour = identify_elements_and_body(image)

    # Extract color and size information
    element_colors, element_sizes, body_color = extract_color_and_size_information(image, element_contours, body_contour)

    # Calculate body size if a contour is found
    if body_contour is not None:
        body_size = cv2.contourArea(body_contour)

        # Calculate balance score
        balance_score = calculate_balance_score(element_sizes, body_size)

        # Calculate proportion score
        proportion_score = calculate_proportion_score(element_sizes, body_size)
    else:
        # If no body contour is found, assign default values
        body_size = 0
        balance_score = 0
        proportion_score = 0

    # Calculate symmetry score
    symmetry_score = calculate_symmetry_score(image)

    # Calculate simplicity score
    simplicity_score = calculate_simplicity_score(element_contours)

    # Calculate harmony score
    harmony_score = calculate_harmony_score(element_colors)

    # Calculate contrast score
    contrast_score = calculate_contrast_score(element_colors, body_color)

    # Calculate unity score
    unity_score = calculate_unity_score(element_sizes)

    # Calculate average aesthetic value
    aesthetic_values = {
        "balance_score": balance_score,
        "proportion_score": proportion_score,
        "symmetry_score": symmetry_score,
        "simplicity_score": simplicity_score,
        "harmony_score": harmony_score,
        "contrast_score": contrast_score,
        "unity_score": unity_score
    }
    avg_aesthetic_value = sum(aesthetic_values.values()) / len(aesthetic_values)

    # Return scores
    return {
        "balance_score": balance_score,
        "proportion_score": proportion_score,
        "symmetry_score": symmetry_score,
        "simplicity_score": simplicity_score,
        "harmony_score": harmony_score,
        "contrast_score": contrast_score,
        "unity_score": unity_score,
        "average_aesthetic_value": avg_aesthetic_value
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    image_data = request.form['image_data']
    scores = process_image(image_data)
    return jsonify(scores)

if __name__ == "__main__":
    app.run(debug=True)
