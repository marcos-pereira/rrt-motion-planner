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
    # Load drawing
    image = cv2.imread(map_name)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get binary threshold
    retval, obstacles_map = cv2.threshold(gray, 10, 255,
                                      cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    obstacles_map = cv2.dilate(obstacles_map, kernel, iterations=1)
    # threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    obstacles_map = cv2.erode(obstacles_map, kernel, iterations=1)
    obstacles_map = cv2.dilate(obstacles_map, kernel, iterations=1)
    obstacles_map = cv2.erode(obstacles_map, kernel, iterations=1)
    obstacles_map = cv2.dilate(obstacles_map, kernel, iterations=1)
    obstacles_map = cv2.erode(obstacles_map, kernel, iterations=1)
    obstacles_map = cv2.dilate(obstacles_map, kernel, iterations=1)
    obstacles_map = cv2.erode(obstacles_map, kernel, iterations=1)
        
    # Find contours of the foreground objects
    contours, _ = cv2.findContours(obstacles_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to remove the background
    mask = np.zeros_like(image)
    
    # Fill the detected foreground contours in the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(image, mask)
    
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