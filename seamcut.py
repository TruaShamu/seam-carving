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

    def energy(self, maskRemoved=None, protectionmask=None):
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
        # just set the energy of the mask to a very high value
        if (protectionmask is not None):
            de_gradient[protectionmask == 1] = 1e9
        return np.sqrt(de_gradient)
    

    def find_horizontal_seam(self, maskRemoved=None, protectionmask=None):
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
        if (protectionmask is not None):
            protectionmask = protectionmask.transpose()
        seam = self.find_vertical_seam(maskRemoved, protectionmask)
        
        # Restore original picture
        self.picture = original_picture
        
        return seam

    def find_vertical_seam(self, maskRemoved=None, protectionmask=None):
        en_mat = self.energy_matrix(maskRemoved, protectionmask)
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
    
    # Assume no additional masking arguments for now
    def find_multiple_vertical_seams(self, num_seams):
        """
        Find multiple seams in the current picture.
        :param num_seams: Number of seams to find.
        :return: A list of lists (each list is a seam).
        """
        # Basically just do the same dp, but get the top num_seams seams with minimum energy, and backtrack for each seam to get list of column indices for each seam
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

        # check the last row and get the minimum num_seams indices
        min_indices = np.argpartition(dp[self.height() - 1], num_seams)[:num_seams]
        #min_index = np.argmin(dp[self.height() - 1])
        #seam = [min_index]

        # Backtrack for each seam
        seams = []
        for min_index in min_indices:
            seam = [min_index]
            for r in range(self.height() - 1, 0, -1):
                min_index = path[r, min_index]
                seam.append(min_index)

            seam.reverse()
            seams.append(seam)
        return seams
    
    def insert_vertical_seams(self, seams):
        """
        Insert multiple vertical seams to the current picture.
        :param seams: A list of lists (each list is a seam).
        """
        for i in range(len(seams)):
            seam = seams[i]
            # get the pixel values of the inserted seam
            inserted_seam_vals = self.vert_inserted_seam_vals(seam)
            # modify self.picture so the seam is duplicated (pixels are inserted at column seam[i]+1 for each row i with the pixel values in inserted_seam_vals)
            self.picture = np.insert(self.picture, seam, inserted_seam_vals, axis=1)
            # the width is increased by 1, so need to add 1 to all the later seams (i.e seams[i+1] to seams[n])
            for j in range(i+1, len(seams)):
                seams[j] = seams[j] + 1
    def vert_inserted_seam_vals(self, seam):
        """
        Given a vertical seam (list of column indices), return the pixel values (RGB) of the inserted seam.
        """
        height, width = self.picture.shape[:2]
        inserted_seam_vals = np.zeros((height, 3), dtype=np.uint8)
        for i in range(height):
            if seam[i] == 0:
                # average itself and the right pixel
                inserted_seam_vals[i] = np.mean(self.picture[i, :2], axis=0)
            elif seam[i] == width - 1:
                # average itself and the left pixel
                inserted_seam_vals[i] = np.mean(self.picture[i, -2:], axis=0)
            else:
                # average itself, left and right pixel
                inserted_seam_vals[i] = np.mean(self.picture[i, seam[i]-1:seam[i]+2], axis=0)
        return inserted_seam_vals
    
    def resize_larger(self, new_width, new_height, seam_batch_size):
        """
        Resize the current picture to the given dimensions.
        :param new_width: New width of the picture.
        :param new_height: New height of the picture.
        :param seam_batch_size: Number of seams to add/remove in each iteration.
        """
        # Calculate the number of seams to add/remove
        vertical_seams = new_width - self.width()
        horizontal_seams = new_height - self.height()
        
        # Add/remove vertical seams
        while vertical_seams > 0:
            # Find multiple vertical seams
            seams = self.find_multiple_vertical_seams(seam_batch_size)
            # Insert the seams
            self.insert_vertical_seams(seams)
            vertical_seams -= seam_batch_size
        
        # Add/remove horizontal seams
        while horizontal_seams > 0:
            # Find multiple horizontal seams
            seams = self.find_multiple_horizontal_seams(seam_batch_size)
            horizontal_seams -= seam_batch_size

    def find_multiple_horizontal_seams(self, num_seams):
        # just transpose the picture and call find_multiple_vertical_seams
        transposed_picture = self.picture.transpose(1, 0, 2)
        self.picture = transposed_picture
        seams = self.find_multiple_vertical_seams(transposed_picture, num_seams)
        # insert the seams
        self.insert_vertical_seams(seams)
        # restore the original picture by transposing it back
        self.picture = self.picture.transpose(1, 0, 2)
    
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

    
    def resize(self, new_width, new_height, maskRemoved=None, protectionmask=None):
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
            seam = self.find_vertical_seam(maskRemoved, protectionmask)
            # draw the seam in red on a copy of the picture
            new_picture = self.draw_vertical_seam(self.picture, seam)
            
            # write the image to {count}.png
            cv2.imwrite(f"{count}.png", new_picture)
            count += 1
            if (maskRemoved is not None):
                maskRemoved = self.remove_vertical_seam(maskRemoved, seam)
            if (protectionmask is not None):
                protectionmask = self.remove_vertical_seam(protectionmask, seam)
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
            if (protectionmask is not None):
                protectionmask = self.remove_horizontal_seam(protectionmask, seam)
            #print("time: ", time.time())
            print("Horizontal cuts remaining: ", horizontal_cuts_remaining)

    
    def get_red(self, x, y):
        return self.picture[y][x][2]
    def get_green(self, x, y):
        return self.picture[y][x][1]
    def get_blue(self, x, y):
        return self.picture[y][x][0]
    def energy_matrix(self, maskRemoved=None, protectionmask=None):
        return self.energy(maskRemoved, protectionmask)
        

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
    

    # deletion mask.
    deletemask = np.load("deletepengin.npy")
    deletemask = deletemask.astype(bool)

    # protection mask
    protectionmask = np.load("protectpengin.npy")
    protectionmask = protectionmask.astype(bool)


    file_path = "2peng.jpg"
    picture = cv2.imread(file_path, cv2.IMREAD_COLOR)

    # Create a seam carver object.
    sc = SeamCarver(picture)

    width= sc.width()
    height = sc.height()
    print("Width: ", width)
    print("Height: ", height)

    # resize to width=400, height=250
    sc.resize(500, 727, deletemask, protectionmask)
    arr = sc.energy_matrix()
    # print the energy matrix formatted

    new_file_path = "new_sample.png"
    cv2.imwrite(new_file_path, sc.picture)

