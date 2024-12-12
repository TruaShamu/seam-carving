import cv2
import numpy as np
import time

count = 0

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

    def energy(self, maskRemoved=None):
        # same as prev but optimize using numpy
        #convert from uint8 to int64 to prevent overflow problems
        arr = np.array(self.picture, dtype = int)
        #calculate squared difference ((x-1, y) - (x+1, y))^2 for each R, G and B pixel
        deltaX2 = np.square(np.roll(arr, -1, axis = 0) - np.roll(arr, 1, axis = 0))
        #same for y axis
        deltaY2 = np.square(np.roll(arr, -1, axis = 1) - np.roll(arr, 1, axis = 1))
        #add R, G and B values for each pixel, then add x- and y-shifted values
        de_gradient = np.sum(deltaX2, axis = 2) + np.sum(deltaY2, axis = 2)
        if (maskRemoved is not None):
            de_gradient[maskRemoved == 1] = 0
        return np.sqrt(de_gradient)
    

    def find_horizontal_seam(self, maskRemoved=None):
        """
        Sequence of indices for the horizontal seam using vertical seam method.
        :return: A list of column indices.
        """
        # Transpose the image to use vertical seam finding
        transposed_picture = self.picture.transpose(1, 0, 2)
        
        # Temporarily replace the picture with transposed version
        original_picture = self.picture
        self.picture = transposed_picture

        # see the transposed picture
        
        # Use existing vertical seam finding method
        
        # if theres a 2d mask, we have to transpose it as well
        if (maskRemoved is not None):
            maskRemoved = maskRemoved.transpose()
        seam = self.find_vertical_seam(maskRemoved)
        
        # Restore original picture
        self.picture = original_picture
        
        return seam

    def find_vertical_seam(self, maskRemoved=None):
        en_mat = self.energy_matrix(maskRemoved)
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
                        
    def draw_vertical_seam(self, picture, seam):
        """
        Draw the vertical seam on the current picture.
        :param seam: List of row indices representing the seam.
        """
        height, width = picture.shape[:2]
        for i in range(height):
            picture[i][seam[i]] = [0, 0, 255]
        return picture
    
    def draw_horizontal_seam(self, picture, seam):
        """
        Draw the horizontal seam on the current picture.
        :param seam: List of column indices representing the seam.
        """
        height, width = picture.shape[:2]
        for i in range(width):
            picture[seam[i]][i] = [0, 0, 255]
        return picture

    """def remove_vertical_seam(self, seam):
        
        Remove the vertical seam from the current picture.
        seam[i] is the column index to be removed at row i.
        
        new_picture = np.zeros((self.height(), self.width() - 1, 3), np.uint8)
        for i in range(self.height()):
            new_picture[i] = np.delete(self.picture[i], seam[i], axis=0)
        self.picture = new_picture"""
    
    '''def remove_vertical_seam(self, input_picture, seam):
        """
        Remove the vertical seam from the current picture efficiently using NumPy.
        seam[i] is the column index to be removed at row i.
        """

        height, width, _ = input_picture.shape
        # Create a mask to remove the specified seam pixels
        mask = np.ones(input_picture.shape[:2], dtype=bool)
        #mask[np.arange(self.height()), seam] = False
        mask[np.arange(height), seam] = False
        # Reshape and select pixels not in the seam
        #new_picture = self.picture[mask].reshape(self.height(), self.width() - 1, 3)
        new_picture = input_picture[mask].reshape(height, width - 1, 3)
        return new_picture
        #self.picture = new_picture
        '''
    
    def remove_vertical_seam(self, input_picture, seam):
        # Get height and width
        height, width = input_picture.shape[:2]

        # Check for 3D (color image) or 2D (grayscale/mask)
        if len(input_picture.shape) == 3:  # For color image
            mask = np.ones((height, width), dtype=bool)
        else:  # For grayscale image or mask
            mask = np.ones((height, width), dtype=bool)

        # Validate seam length matches height
        if len(seam) != height:
            raise ValueError(f"Seam length {len(seam)} does not match image height {height}.")

        # Set seam positions to False in the mask
        mask[np.arange(height), seam] = False

        # Apply mask to remove the seam
        if len(input_picture.shape) == 3:  # For color image
            output_picture = input_picture[mask].reshape((height, width - 1, input_picture.shape[2]))
        else:  # For grayscale image or mask
            output_picture = input_picture[mask].reshape((height, width - 1))

        return output_picture

            

    def remove_horizontal_seam(self, input_picture, seam):
        # dimensions of the original image
        height, width = input_picture.shape[:2]
        print("input mask when removing horizontal seam: ", input_picture.shape)
        print("height: ", height)
        print("width: ", width)
        # Transpose the image to treat horizontal seam as a vertical one
        transposed_picture = np.transpose(input_picture, (1, 0, 2)) if len(input_picture.shape) == 3 else np.transpose(input_picture)
        transposed_seam = seam  # Seam corresponds to rows in the transposed image

        # Remove the seam using the optimized vertical seam function
        new_picture = self.remove_vertical_seam(transposed_picture, transposed_seam)

        # Transpose back to the original orientation
        return np.transpose(new_picture, (1, 0, 2)) if len(input_picture.shape) == 3 else np.transpose(new_picture)

    
    def resize(self, new_width, new_height, maskRemoved=None):
        """
        Resize the current picture to the given dimensions.
        :param new_width: New width of the picture.
        :param new_height: New height of the picture.
        :param maskRemoved: A 2D list representing the mask to remove (optional)
        """
        global count
        horizontal_cuts_remaining = self.height() - new_height
        vertical_cuts_remaining = self.width() - new_width
        #print("time: ", time.time())

        while vertical_cuts_remaining >0:
            #if (maskRemoved is not None):
            #    en_mat[maskRemoved == 1] = 0
            seam = self.find_vertical_seam(maskRemoved)
            # draw the seam in red on a copy of the picture
            new_picture = self.draw_vertical_seam(self.picture, seam)
            
            # write the image to {count}.png
            cv2.imwrite(f"{count}.png", new_picture)
            count += 1
            if (maskRemoved is not None):
                maskRemoved = self.remove_vertical_seam(maskRemoved, seam)
            vertical_cuts_remaining -= 1
            #print("time: ", time.time())
            print("Vertical cuts remaining: ", vertical_cuts_remaining)
            self.picture = self.remove_vertical_seam(self.picture, seam)
        
        while horizontal_cuts_remaining > 0:
            #if (maskRemoved is not None):
            #    en_mat[maskRemoved == 1] = 0
            seam = self.find_horizontal_seam(maskRemoved)
            horizontal_cuts_remaining -= 1
            self.remove_horizontal_seam(self.picture, seam)
            # we also have to remove the seam from the mask
            if (maskRemoved is not None):
                maskRemoved = self.remove_horizontal_seam(maskRemoved, seam)
            #print("time: ", time.time())
            print("Horizontal cuts remaining: ", horizontal_cuts_remaining)

    
    def get_red(self, x, y):
        return self.picture[y][x][2]
    def get_green(self, x, y):
        return self.picture[y][x][1]
    def get_blue(self, x, y):
        return self.picture[y][x][0]
    def energy_matrix(self, maskRemoved=None):
        return self.energy(maskRemoved)
        

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
    count = 0
    print("time: ", time.time())

    # Read an image from local filesys.
    mask = np.load("surfer.npy")
    file_path = "surfer.jpg"
    picture = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # to double check, let's overlay the mask on the image
    mask = mask.astype(bool)
    #picture[mask == 1] = [0, 0, 255]
    #cv2.imshow("Masked Image", picture)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Create a seam carver object.
    sc = SeamCarver(picture)

    # Get the width and height of the picture.
    width = sc.width()
    height = sc.height()
    print("Width: ", width)
    print("Height: ", height)

    # print the size of the mask
    print("mask width: ", mask.shape[1])
    print("mask height: ", mask.shape[0])


    # resize to width=400, height=250
    sc.resize(1400, 1079, mask)
    arr = sc.energy_matrix()
    # print the energy matrix formatted

    new_file_path = "new_sample.png"
    cv2.imwrite(new_file_path, sc.picture)
    print("time: ", time.time())
    print("we done")

    # Visualize the energy matrix
    en_mat = sc.energy_matrix()

    # fill 0s in the mask
    en_mat[mask == 1] = 0
    visualize_energy(en_mat)

    # Find and draw the vertical seam
    #seam = sc.find_vertical_seam(en_mat)
    #sc.draw_vertical_seam(seam)

    seam = sc.find_horizontal_seam()
    sc.draw_horizontal_seam(seam)

    # draw the mask on the picture in green
    sc.picture[mask == 1] = [0, 255, 0]

    # try to remove the vertical seam from the mask
    mask = sc.remove_horizontal_seam(mask, seam)

    # render the image
    cv2.imshow("Vertical Seam", sc.picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

