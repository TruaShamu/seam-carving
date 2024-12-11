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
        Sequence of indices for the horizontal seam.
        :return: A list of column indices.
        """
        en_mat = self.energy_matrix()
        # dp[i][j] = minimum energy of seam ending at (i, j)
        dp = np.zeros((self.height(), self.width()), dtype=np.float32)

        # make a matrix to contain the path. path[i][j] = r) means that the seam ending at (i, j) came from row r, column c-1
        path = np.zeros((self.height(), self.width(), 2), dtype=np.int32)

        # initialize the first column
        dp[:, 0] = en_mat[:, 0]

        # fill the dp table
        for c in range(1, self.width()):
            for r in range(self.height()):
                up = r - 1 if r > 0 else r
                down = r + 1 if r < self.height() - 1 else r

                prevcost = min(dp[up][c - 1], dp[r][c - 1], dp[down][c - 1])
                dp[r][c] = en_mat[r][c] + prevcost

                if (prevcost == dp[up][c - 1]):
                    path[r][c] = up
                elif (prevcost == dp[r][c - 1]):
                    path[r][c] = r
                else:
                    path[r][c] = down
        
        # find the minimum cost in the last column
        min_cost = min([dp[i][self.width() - 1] for i in range(self.height())])

        # backtrack to find the seam
        seam = []
        
        # check last column
        min_index = np.argmin(dp[:, self.width() - 1])

        seam.append(min_index)
        for c in range(self.width() - 1, 0, -1):
            min_index = path[min_index][c]
            seam.append(min_index)
        seam.reverse()
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

    def remove_vertical_seam(self, seam):
        """
        Remove the vertical seam from the current picture.
        seam[i] is the column index to be removed at row i.
        """
        new_picture = np.zeros((self.height(), self.width() - 1, 3), np.uint8)
        for i in range(self.height()):
            new_picture[i] = np.delete(self.picture[i], seam[i], axis=0)
        self.picture = new_picture

    def remove_horizontal_seam(self, seam):
        """
        Remove the horizontal seam from the current picture.
        seam[i] is the row index to be removed at column i.
        """
        new_picture = np.zeros((self.height() - 1, self.width(), 3), np.uint8)
        for i in range(self.width()):
            new_picture[:, i] = np.delete(self.picture[:, i], seam[i], axis=0)
        self.picture = new_picture

    
    def resize(self, new_width, new_height):
        """
        Resize the current picture to the given dimensions.
        :param new_width: New width of the picture.
        :param new_height: New height of the picture.
        """
        horizontal_cuts_remaining = self.height() - new_height
        vertical_cuts_remaining = self.width() - new_width
        print("time: ", time.time())

        while vertical_cuts_remaining >0:
            seam = self.find_vertical_seam()
            print("time: ", time.time())
            vertical_cuts_remaining -= 1
            print("Vertical cuts remaining: ", vertical_cuts_remaining)
            self.remove_vertical_seam(seam)
        
        while horizontal_cuts_remaining > 0:
            seam = self.find_horizontal_seam()
            horizontal_cuts_remaining -= 1
            self.remove_horizontal_seam(seam)
            print("time: ", time.time())
            print("Horizontal cuts remaining: ", horizontal_cuts_remaining)

    
    def get_red(self, x, y):
        return self.picture[y][x][2]
    def get_green(self, x, y):
        return self.picture[y][x][1]
    def get_blue(self, x, y):
        return self.picture[y][x][0]
    def energy_matrix(self):
        return self.energy()
        
'''
    def energy_matrix(self):
        """
        Compute the energy matrix using the Sobel operator.
        :return: A 2D energy matrix.
        """
        return self.energy_function()
'''
# Unit testing (required)
if __name__ == "__main__":

    # Read an image from local filesys.
    file_path = "2peng.jpg"
    picture = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Create a seam carver object.
    sc = SeamCarver(picture)

    # Get the width and height of the picture.
    width = sc.width()
    height = sc.height()
    print("Width: ", width)
    print("Height: ", height)

    # resize to width=400, height=250
    sc.resize(727, 727)
    arr = sc.energy_matrix()
    # print the energy matrix formatted

    new_file_path = "new_sample.png"
    cv2.imwrite(new_file_path, sc.picture)