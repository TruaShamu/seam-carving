import cv2
import numpy as np
import time


class SeamCarver:
    def __init__(self, picture):
        """
        Create a seam carver object based on the given picture.
        :param picture: A 2D list or any image representation.
        """
        self.picture = np.array(picture)


    def width(self):
        return self.picture.shape[1]

    def height(self):
        """
        Height of the current picture.
        :return: Height as an integer.
        """
        return self.picture.shape[0]

    def energy(self):
        # same as prev but optimize using numpy
        #convert from uint8 to int64 to prevent overflow problems
        arr = np.array(self.picture, dtype = int)
        #calculate squared difference ((x-1, y) - (x+1, y))^2 for each R, G and B pixel
        deltaX2 = np.square(np.roll(arr, -1, axis = 0) - np.roll(arr, 1, axis = 0))
        #same for y axis
        deltaY2 = np.square(np.roll(arr, -1, axis = 1) - np.roll(arr, 1, axis = 1))
        #add R, G and B values for each pixel, then add x- and y-shifted values
        de_gradient = np.sum(deltaX2, axis = 2) + np.sum(deltaY2, axis = 2)
        return np.sqrt(de_gradient)
    

    def find_horizontal_seam(self):
        """
        Sequence of indices for the horizontal seam using vertical seam method.
        :return: A list of column indices.
        """
        # Transpose the image to use vertical seam finding
        transposed_picture = self.picture.transpose(1, 0, 2)
        
        # Temporarily replace the picture with transposed version
        original_picture = self.picture
        self.picture = transposed_picture
        
        # Use existing vertical seam finding method
        seam = self.find_vertical_seam()
        
        # Restore original picture
        self.picture = original_picture
        
        return seam

    def find_vertical_seam(self):
        en_mat = self.energy_matrix()
        dp = np.zeros((self.height(), self.width()), dtype=np.float32)
        path = np.zeros((self.height(), self.width()), dtype=np.int32)  # Only store column indices

        dp[0, :] = en_mat[0, :]

        # Fill the dp table
        for r in range(1, self.height()):
            left = np.roll(dp[r - 1], 1)
            center = dp[r - 1]
            right = np.roll(dp[r - 1], -1)

            left[0] = float('inf')
            right[-1] = float('inf')

            prevcost = np.minimum(np.minimum(left, center), right)
            dp[r] = en_mat[r] + prevcost

            path[r] = np.where(prevcost == left, np.arange(self.width()) - 1,
                    np.where(prevcost == center, np.arange(self.width()),
                            np.arange(self.width()) + 1))

        # Find the minimum cost in the last row
        min_index = np.argmin(dp[self.height() - 1])
        seam = [min_index]

        # Backtrack
        for r in range(self.height() - 1, 0, -1):
            min_index = path[r, min_index]
            seam.append(min_index)

        seam.reverse()
        return seam



                        
    def draw_vertical_seam(self, seam):
        """
        Draw the vertical seam on the current picture.
        :param seam: List of row indices representing the seam.
        """
        for i in range(self.height()):
            self.picture[i][seam[i]] = [0, 0, 255]
    
    def draw_horizontal_seam(self, seam):
        """
        Draw the horizontal seam on the current picture.
        :param seam: List of column indices representing the seam.
        """
        for i in range(self.width()):
            self.picture[seam[i]][i] = [0, 0, 255]

    """def remove_vertical_seam(self, seam):
        
        Remove the vertical seam from the current picture.
        seam[i] is the column index to be removed at row i.
        
        new_picture = np.zeros((self.height(), self.width() - 1, 3), np.uint8)
        for i in range(self.height()):
            new_picture[i] = np.delete(self.picture[i], seam[i], axis=0)
        self.picture = new_picture"""
    
    def remove_vertical_seam(self, seam):
        """
        Remove the vertical seam from the current picture efficiently using NumPy.
        seam[i] is the column index to be removed at row i.
        """
        # Create a mask to remove the specified seam pixels
        mask = np.ones(self.picture.shape[:2], dtype=bool)
        mask[np.arange(self.height()), seam] = False
        
        # Reshape and select pixels not in the seam
        new_picture = self.picture[mask].reshape(self.height(), self.width() - 1, 3)
        
        self.picture = new_picture
        

    def remove_horizontal_seam(self, seam):
        """
        Remove the horizontal seam from the current picture using transposition.
        seam[i] is the row index to be removed at column i.
        """
        # Transpose the image
        transposed_picture = self.picture.transpose(1, 0, 2)
        
        # Set the picture to transposed version
        self.picture = transposed_picture
        
        # Use vertical seam removal
        self.remove_vertical_seam(seam)
        
        # Transpose back
        self.picture = self.picture.transpose(1, 0, 2)
    
    def resize(self, new_width, new_height):
        """
        Resize the current picture to the given dimensions.
        :param new_width: New width of the picture.
        :param new_height: New height of the picture.
        """
        horizontal_cuts_remaining = self.height() - new_height
        vertical_cuts_remaining = self.width() - new_width
        #print("time: ", time.time())

        while vertical_cuts_remaining >0:
            seam = self.find_vertical_seam()
            vertical_cuts_remaining -= 1
            #print("time: ", time.time())
            #print("Vertical cuts remaining: ", vertical_cuts_remaining)
            self.remove_vertical_seam(seam)
        
        while horizontal_cuts_remaining > 0:
            seam = self.find_horizontal_seam()
            horizontal_cuts_remaining -= 1
            self.remove_horizontal_seam(seam)
            #print("time: ", time.time())
            #print("Horizontal cuts remaining: ", horizontal_cuts_remaining)

    
    def get_red(self, x, y):
        return self.picture[y][x][2]
    def get_green(self, x, y):
        return self.picture[y][x][1]
    def get_blue(self, x, y):
        return self.picture[y][x][0]
    def energy_matrix(self):
        return self.energy()
        

# Visualize the energy matrix using opencv
def visualize_energy(energy_matrix):
    """
    Visualize the energy matrix using OpenCV.
    :param energy_matrix: A 2D list representing the energy matrix.
    """
    # Normalize the energy matrix
    energy_matrix = (energy_matrix - np.min(energy_matrix)) / (np.max(energy_matrix) - np.min(energy_matrix)) * 255
    energy_matrix = energy_matrix.astype(np.uint8)
    
    # Create a grayscale image
    energy_image = cv2.cvtColor(energy_matrix, cv2.COLOR_GRAY2BGR)
    
    # Display the image
    cv2.imshow("Energy Matrix", energy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Unit testing (required)
if __name__ == "__main__":
    print("time: ", time.time())

    # Read an image from local filesys.
    file_path = "surfer.jpg"
    picture = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Create a seam carver object.
    sc = SeamCarver(picture)

    # Get the width and height of the picture.
    width = sc.width()
    height = sc.height()
    print("Width: ", width)
    print("Height: ", height)

    # resize to width=400, height=250
    sc.resize(1500, 1079)
    #arr = sc.energy_matrix()
    # print the energy matrix formatted

    new_file_path = "new_sample.png"
    cv2.imwrite(new_file_path, sc.picture)
    print("time: ", time.time())

    # Visualize the energy matrix
    visualize_energy(sc.energy_matrix())