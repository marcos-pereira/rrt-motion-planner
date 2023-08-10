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
    retval, threshold = cv2.threshold(gray, 10, 255,
                                      cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    # threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    threshold = cv2.erode(threshold, kernel, iterations=1)
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    threshold = cv2.erode(threshold, kernel, iterations=1)
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    threshold = cv2.erode(threshold, kernel, iterations=1)
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    threshold = cv2.erode(threshold, kernel, iterations=1)

    if not test:
        cv2.imshow("drawing", threshold)
        cv2.waitKey(0)

    ## Get binary matrix from threshold image
    rows, cols = threshold.shape
    drawing_matrix = np.empty([rows, cols])
    for row in range(rows):
        for col in range(cols):
            if threshold[row, col] == 255:
                drawing_matrix[row, col] = 1
            if threshold[row, col] == 0:
                drawing_matrix[row, col] = 0

    ## Get indices of coordinates with 1's in the drawing matrix
    ones_in_drawing = np.where(drawing_matrix == 1)
    drawing_coordinates = list(zip(ones_in_drawing[0], ones_in_drawing[1]))

    return drawing_matrix