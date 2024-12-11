import cv2
import numpy as np

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

    # dual gradient energy function diy impl
    """
    def energy(self, x, y):
        if x == 0 or x == self.width() - 1 or y == 0 or y == self.height() - 1:
            return 1000.0
        
        neighbors = self.picture[y-1:y+2, x-1:x+2]
        dx = np.diff(neighbors[:, :, :], axis=1)
        dy = np.diff(neighbors[:, :, :], axis=0)
        
        x_gradient_sq = np.sum(dx[0, 0, :]**2)
        y_gradient_sq = np.sum(dy[0, 0, :]**2)
        
        return np.sqrt(x_gradient_sq + y_gradient_sq)
    """

    def energy_function(self):
        """
        Compute the energy map of the image using the Sobel operator.
        :param image: The input image as a numpy array.
        :return: A 2D energy map.
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.picture, cv2.COLOR_BGR2GRAY)
        
        # Compute the gradients in x and y directions
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in x-direction
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in y-direction
        
        # Compute the energy as the magnitude of the gradient
        energy = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize the energy map for visualization (optional)
        energy = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX)
        
        return energy
    def find_horizontal_seam(self):
        """
        Sequence of indices for the horizontal seam.
        :return: A list of column indices.
        """
        en_mat = self.energy_matrix()
        # dp[i][j] = minimum energy of seam ending at (i, j)
        dp = [[0 for _ in range(self.width())] for _ in range(self.height())]

        # make a matrix to contain the path. path[i][j] = r) means that the seam ending at (i, j) came from row r, column c-1
        path = [[(0, 0) for _ in range(self.width())] for _ in range(self.height())]

        # initialize the first column
        for i in range(self.height()):
            dp[i][0] = en_mat[i][0]

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
        min_index = [dp[i][self.width() - 1] for i in range(self.height())].index(min_cost)

        seam.append(min_index)
        for c in range(self.width() - 1, 0, -1):
            min_index = path[min_index][c]
            seam.append(min_index)
        seam.reverse()
        return seam

    def find_vertical_seam(self):
        """
        Sequence of indices for the vertical seam.
        :return: A list of row indices.
        """
        en_mat = self.energy_matrix()
        # dp[i][j] = minimum energy of seam ending at (i, j)
        dp = [[0 for _ in range(self.width())] for _ in range(self.height())]

        # make a matrix to contain the path. path[i][j] = c) means that the seam ending at (i, j) came from row r-1, column c
        path = [[(0, 0) for _ in range(self.width())] for _ in range(self.height())]

        # initialize the top row
        for i in range(self.width()):
            dp[0][i] = en_mat[0][i]
        
        # fill the dp table
        for r in range(1, self.height()):
            for c in range(self.width()):
                left = c - 1 if c > 0 else c
                right = c + 1 if c < self.width() - 1 else c

                prevcost = min(dp[r - 1][left], dp[r - 1][c], dp[r - 1][right])
                dp[r][c] = en_mat[r][c] + prevcost

                if (prevcost == dp[r - 1][left]):
                    path[r][c] = left
                elif (prevcost == dp[r - 1][c]):
                    path[r][c] = c
                else:
                    path[r][c] = right
        
        # find the minimum cost in the last row
        min_cost = min(dp[self.height() - 1])

        # backtrack to find the seam
        seam = []
        min_index = dp[self.height() - 1].index(min_cost)
        seam.append(min_index)
        for r in range(self.height() - 1, 0, -1):
            min_index = path[r][min_index]
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

        while vertical_cuts_remaining >0:
            seam = self.find_vertical_seam()
            vertical_cuts_remaining -= 1
            print("Vertical cuts remaining: ", vertical_cuts_remaining)
            self.remove_vertical_seam(seam)
        
        while horizontal_cuts_remaining > 0:
            seam = self.find_horizontal_seam()
            horizontal_cuts_remaining -= 1
            self.remove_horizontal_seam(seam)
            print("Horizontal cuts remaining: ", horizontal_cuts_remaining)

    
    def get_red(self, x, y):
        return self.picture[y][x][2]
    def get_green(self, x, y):
        return self.picture[y][x][1]
    def get_blue(self, x, y):
        return self.picture[y][x][0]
    '''def energy_matrix(self):
        return [[self.energy(x, y) for x in range(self.width())] for y in range(self.height())]
    '''

    def energy_matrix(self):
        """
        Compute the energy matrix using the Sobel operator.
        :return: A 2D energy matrix.
        """
        return self.energy_function()

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
    sc.resize(950, 710)

    # Save the new image to local filesys.
    new_file_path = "new_sample.png"
    cv2.imwrite(new_file_path, sc.picture)