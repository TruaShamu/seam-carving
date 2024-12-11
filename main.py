import cv2

class SeamCarver:
    def __init__(self, picture):
        """
        Create a seam carver object based on the given picture.
        :param picture: A 2D list or any image representation.
        """
        self.picture = picture

    def picture(self):
        """
        Current picture.
        :return: The current picture.
        """
        return self.picture

    def width(self):
        return self.picture.shape[1]

    def height(self):
        """
        Height of the current picture.
        :return: Height as an integer.
        """
        return self.picture.shape[0]

    def energy(self, x, y):
        """
        Energy of the pixel at column x and row y.
        :param x: Column index.
        :param y: Row index.
        :return: Energy as a float.
        """
        # This function would calculate the energy of a pixel based on some criteria.

        # Border cases: Energy of pixels on the border would be 1000.
        if x == 0 or x == self.width() - 1 or y == 0 or y == self.height() - 1:
            return 1000.0
        
        # dual-gradient energy function

        rx = int(self.get_red(x + 1, y)) - int(self.get_red(x - 1, y))
        gx = int(self.get_green(x + 1, y)) - int(self.get_green(x - 1, y))
        bx = int(self.get_blue(x + 1, y)) - int(self.get_blue(x - 1, y))
        x_gradient_sq = rx * rx + gx * gx + bx * bx

        ry = int(self.get_red(x, y + 1)) - int(self.get_red(x, y - 1))
        gy = int(self.get_green(x, y + 1)) - int(self.get_green(x, y - 1))
        by = int(self.get_blue(x, y + 1)) - int(self.get_blue(x, y - 1))
        y_gradient_sq = ry * ry + gy * gy + by * by

        return (x_gradient_sq + y_gradient_sq) ** 0.5

    def find_horizontal_seam(self):
        """
        Sequence of indices for the horizontal seam.
        :return: A list of column indices.
        """


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
                
            

    def remove_horizontal_seam(self, seam):
        """
        Remove horizontal seam from the current picture.
        :param seam: List of column indices representing the seam.
        """
        raise NotImplementedError("Remove horizontal seam not implemented.")

    def remove_vertical_seam(self, seam):
        """
        Remove vertical seam from the current picture.
        :param seam: List of row indices representing the seam.
        """
        raise NotImplementedError("Remove vertical seam not implemented.")
    
    def get_red(self, x, y):
        #print("red: ", self.picture[y][x][2])
        return self.picture[y][x][2]
    def get_green(self, x, y):
        #print("green: ", self.picture[y][x][1])
        return self.picture[y][x][1]
    def get_blue(self, x, y):
        #print("blue: ", self.picture[y][x][0])
        return self.picture[y][x][0]
    def energy_matrix(self):
        return [[self.energy(x, y) for x in range(self.width())] for y in range(self.height())]

# Unit testing (required)
if __name__ == "__main__":
    # Example usage (would require an actual implementation and picture input)

    # Read an image from local filesys.
    file_path = "sample.png"
    picture = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Create a seam carver object.
    sc = SeamCarver(picture)

    # Get the width and height of the picture.
    width = sc.width()
    height = sc.height()
    print("Width: ", width)
    print("Height: ", height)

    # Get the energy matrix of the picture.
    energy_matrix = sc.energy_matrix()

    # Print the energy matrix
    #for row in energy_matrix:
    #    print(row)
    
    # Find the vertical seam.
    vertical_seam = sc.find_vertical_seam()

    # Visualize the vertical seam on the image
    for i in range(height):
        picture[i][vertical_seam[i]] = [0, 0, 255]
    
    # Save the image with the seam visualized.
    cv2.imwrite("sample_seam.jpg", picture)
    
