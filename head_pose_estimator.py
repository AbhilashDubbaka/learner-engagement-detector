import cv2
import numpy as np

class Headpose_Estimator():

    def __init__(self):
        # 3D facial model coordinates
        self.landmarks_3d = np.array([
            [ 0.0,  0.0,   0.0],
            [ 0.0, -330.0,  -65.0],
            [-225.0,  170.0,  -135.0],
            [ 225.0,  170.0,  -135.0],
            [-150.0, -150.0,  -125.0],
            [ 150.0, -150.0,  -125.0]
        ], dtype=np.double)

        # 2d facial landmarks
        self.lm_2d_index = [30, 8, 36, 45, 48, 54]

    def get_headpose(self, im, landmarks_2d):
        h, w, c = im.shape
        f = w
        x, y = w / 2, h / 2
        camera_matrix = np.array(
            [[f, 0, x],
             [0, f, y],
             [0, 0, 1]], dtype = np.double
         )

        dist_coeffs = np.zeros((4,1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.landmarks_3d, landmarks_2d, camera_matrix, dist_coeffs)

        rvec_mat = cv2.Rodrigues(rotation_vector)[0]
        proj_mat = np.hstack((rvec_mat, translation_vector))
        degrees = -cv2.decomposeProjectionMatrix(proj_mat)[6]
        x, y, z = degrees[:, 0]

        return [x, y, z]

    def process_image(self, im, shape):
        landmarks = []
        for i in self.lm_2d_index:
            landmarks += [[shape.part(i).x, shape.part(i).y]]
        landmarks_2d = (np.array(landmarks).astype(np.double))
        
        angles = self.get_headpose(im, landmarks_2d)
        x, y, z = angles
        if x > 0:
            x = x - 180
        else:
            x = x + 180
        angles[0] = x

        return angles