#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

#!/usr/bin/env python
import cv2
import numpy as np

def load_map(map_name, test=False):
    """Return a binary matrix where 0 indicate free space and 1 
    indicate occupied space by obstacle.

    Args:
        map_name (_type_): the name of the map figure as a .png file.
        test (bool, optional): if in test mode, i.e., True, the map is not shown. 
        Defaults to False.

    Returns:
        numpy array: the binary matrix map where 0 indicte free space and
        1 indicate occupied space by obstacle.
    """
    # Load drawing
    image = cv2.imread(map_name)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get binary threshold
    retval, obstacles_map = cv2.threshold(gray, 10, 255,
                                      cv2.THRESH_BINARY)

    # Morphological operations
    # kernel = np.ones((3, 3), np.uint8)
    # obstacles_map = cv2.dilate(obstacles_map, kernel, iterations=1)
    # # threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    # obstacles_map = cv2.erode(obstacles_map, kernel, iterations=1)
    # obstacles_map = cv2.dilate(obstacles_map, kernel, iterations=1)
    # obstacles_map = cv2.erode(obstacles_map, kernel, iterations=1)
    # obstacles_map = cv2.dilate(obstacles_map, kernel, iterations=1)
    # obstacles_map = cv2.erode(obstacles_map, kernel, iterations=1)
    # obstacles_map = cv2.dilate(obstacles_map, kernel, iterations=1)
    # obstacles_map = cv2.erode(obstacles_map, kernel, iterations=1)
    
    # Invert black and white
    no_background_image = 255 - obstacles_map
    
    # Apply adaptive thresholding to create a binary mask
    _, thresholded = cv2.threshold(no_background_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=3)
        
    # Find contours in the mask
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the foreground
    foreground_mask = np.zeros_like(opening)
    cv2.drawContours(foreground_mask, contours, -1, 255, thickness=cv2.FILLED)

    background_color= (3, 171, 173)
    bgr_color = (background_color[2], background_color[1], background_color[0])

    # Create a white background
    background = np.ones_like(image) * bgr_color

    # Copy the foreground onto the white background using the mask
    result = np.where(np.expand_dims(foreground_mask, axis=-1), image, background)

    # Save the result to the specified output path
    map_no_background = "no_background.png"
    cv2.imwrite(map_no_background, result)

    if not test:
        cv2.imshow("drawing", obstacles_map)
        cv2.waitKey(0)

    obstacles_map = 255 - obstacles_map

    ## Get binary matrix from threshold image
    rows, cols = obstacles_map.shape
    drawing_matrix = np.empty([rows, cols])
    for row in range(rows):
        for col in range(cols):
            if obstacles_map[row, col] == 255:
                drawing_matrix[row, col] = 1
            if obstacles_map[row, col] == 0:
                drawing_matrix[row, col] = 0

    ## Get indices of coordinates with 1's in the drawing matrix
    ones_in_drawing = np.where(drawing_matrix == 1)
    drawing_coordinates = list(zip(ones_in_drawing[0], ones_in_drawing[1]))

    return drawing_matrix