import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def start():
    path_to_img = 'Images/I4/'

    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images

    images = load_images_from_folder(path_to_img)

    file_names = []
    for i in range(len(images)):
        file_names.append(os.listdir(path_to_img)[i])

    print(file_names)
    file_names = sorted(file_names)
    print(file_names)

    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for i in range(len(images)):
        print("Processing image: ", i)
        image = cv2.imread(path_to_img + file_names[i])
        kp, des = sift.detectAndCompute(image, None)
        keypoints.append(kp)
        descriptors.append(des)
        image_with_sift = cv2.drawKeypoints(image, kp, None)
        # plt.figure()
        # plt.imshow(cv2.cvtColor(image_with_sift, cv2.COLOR_BGR2RGB))
        # plt.title('SIFT Features in Image {}'.format(file_names[i]))
        # plt.show()


    bf = cv2.BFMatcher()

    matches = []
    for i in range(len(images) - 1):
        print("Matching image: ", file_names[i], " with image: ", file_names[i+1])
        matches.append(bf.knnMatch(descriptors[i], descriptors[i+1], k=2))

    for i in range(len(matches)):
        print("Plotting matches between image: ", file_names[i], " and image: ", file_names[i+1])
        image1 = cv2.imread(path_to_img + file_names[i])
        image2 = cv2.imread(path_to_img + file_names[i+1])
        matches_image = cv2.drawMatchesKnn(image1, keypoints[i], image2, keypoints[i+1], matches[i], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.figure()
        # plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB))
        # plt.title('Matches between Image {} and Image {}'.format(file_names[i], file_names[i+1]))
        # plt.show()

    homography_matrices_scratch = []

    for i in range(len(matches)):
        print("Calculating homography matrix between image: ", file_names[i], " and image: ", file_names[i+1])
        good_matches = []
        for m, n in matches[i]:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i+1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        N = 2000
        T = 5
        M = len(good_matches)
        n = 4
        H_best = np.zeros((3, 3))
        max_inliers = 0
        for j in tqdm(range(N)):
            random_indices = np.random.choice(M, n, replace=False)
            src_pts_random = np.float32([src_pts[i] for i in random_indices]).reshape(-1, 1, 2)
            dst_pts_random = np.float32([dst_pts[i] for i in random_indices]).reshape(-1, 1, 2)
            A = np.zeros((2*n, 9))
            for k in range(n):
                x = src_pts_random[k][0][0]
                y = src_pts_random[k][0][1]
                x_prime = dst_pts_random[k][0][0]
                y_prime = dst_pts_random[k][0][1]
                A[2*k] = [-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime]
                A[2*k+1] = [0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime]
            U, S, V = np.linalg.svd(A)
            H = V[-1].reshape(3, 3)

            inliers = 0
            for k in range(M):
                src_pt = np.array([src_pts[k][0][0], src_pts[k][0][1], 1])
                dst_pt = np.array([dst_pts[k][0][0], dst_pts[k][0][1], 1])
                dst_pt_transformed = np.dot(H, src_pt)
                dst_pt_transformed = dst_pt_transformed / dst_pt_transformed[2]
                if np.linalg.norm(dst_pt - dst_pt_transformed) < T:
                    inliers += 1
            if inliers > max_inliers:
                max_inliers = inliers
                H_best = H
        homography_matrices_scratch.append(H_best)
        print("The Homography Matrix is: ", H_best)

    homography_matrices = homography_matrices_scratch

    homo_01 = homography_matrices[0]
    homo_10 = np.linalg.inv(homo_01)

    homo_12 = homography_matrices[1]
    homo_21 = np.linalg.inv(homo_12)

    homo_23 = homography_matrices[2]
    homo_32 = np.linalg.inv(homo_23)

    homo_34 = homography_matrices[3]
    homo_43 = np.linalg.inv(homo_34)


    homo_02 = np.dot(homo_01, homo_12)
    homo_12 = homo_12
    homo_22 = np.eye(2,2)
    homo_32 = homo_32
    homo_42 = np.dot(homo_43, homo_32)


    homo_00 = np.eye(3) 
    homo_10 = homo_10
    homo_20 = np.dot(homo_21, homo_10)
    homo_30 = np.dot(homo_32, homo_20)
    homo_40 = np.dot(homo_43, homo_30)


    height, width, channels = images[0].shape


    corners = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32).reshape(-1, 1, 2)
    
    def apply_homography(corners, H):
        transformed_corners = []
        x, y, z = corners.shape
        corners = corners.reshape(x, z)
        for corner in corners:
            point = np.array([corner[0], corner[1], 1.0])
            transformed_point = np.dot(H, point)
            transformed_point /= transformed_point[2]  # Normalize
            transformed_corners.append((transformed_point[0], transformed_point[1]))
        transformed_corners = np.array(transformed_corners)
        x, z = transformed_corners.shape
        transformed_corners = transformed_corners.reshape(x, 1, z)
        return transformed_corners

    transformed_corners_0 = apply_homography(corners, homo_02)
    transformed_corners_1 = apply_homography(corners, homo_12)
    # transformed_corners_2 = apply_homography(corners, homo_22)
    transformed_corners_3 = apply_homography(corners, homo_32)
    transformed_corners_4 = apply_homography(corners, homo_42)


    image0_corners = transformed_corners_0
    image1_corners = transformed_corners_1
    image2_corners = corners
    image3_corners = transformed_corners_3
    image4_corners = transformed_corners_4


    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0

    for i in range(4):
        if image0_corners[i][0][0] < min_x:
            min_x = image0_corners[i][0][0]
        if image0_corners[i][0][1] < min_y:
            min_y = image0_corners[i][0][1]
        if image0_corners[i][0][0] > max_x:
            max_x = image0_corners[i][0][0]
        if image0_corners[i][0][1] > max_y:
            max_y = image0_corners[i][0][1]

        if image1_corners[i][0][0] < min_x:
            min_x = image1_corners[i][0][0]
        if image1_corners[i][0][1] < min_y:
            min_y = image1_corners[i][0][1]
        if image1_corners[i][0][0] > max_x:
            max_x = image1_corners[i][0][0]
        if image1_corners[i][0][1] > max_y:
            max_y = image1_corners[i][0][1]

        if image2_corners[i][0][0] < min_x:
            min_x = image2_corners[i][0][0]
        if image2_corners[i][0][1] < min_y:
            min_y = image2_corners[i][0][1]
        if image2_corners[i][0][0] > max_x:
            max_x = image2_corners[i][0][0]
        if image2_corners[i][0][1] > max_y:
            max_y = image2_corners[i][0][1]

        if image3_corners[i][0][0] < min_x:
            min_x = image3_corners[i][0][0]
        if image3_corners[i][0][1] < min_y:
            min_y = image3_corners[i][0][1]
        if image3_corners[i][0][0] > max_x:
            max_x = image3_corners[i][0][0]
        if image3_corners[i][0][1] > max_y:
            max_y = image3_corners[i][0][1]

        if image4_corners[i][0][0] < min_x:
            min_x = image4_corners[i][0][0]
        if image4_corners[i][0][1] < min_y:
            min_y = image4_corners[i][0][1]
        if image4_corners[i][0][0] > max_x:
            max_x = image4_corners[i][0][0]
        if image4_corners[i][0][1] > max_y:
            max_y = image4_corners[i][0][1]


    print("Minimum x: ", min_x)
    print("Minimum y: ", min_y)
    print("Maximum x: ", max_x)
    print("Maximum y: ", max_y)

    for i in range(4):
        image0_corners[i][0][0] = image0_corners[i][0][0] - min_x
        image0_corners[i][0][1] = image0_corners[i][0][1] - min_y
        image1_corners[i][0][0] = image1_corners[i][0][0] - min_x
        image1_corners[i][0][1] = image1_corners[i][0][1] - min_y
        image2_corners[i][0][0] = image2_corners[i][0][0] - min_x
        image2_corners[i][0][1] = image2_corners[i][0][1] - min_y
        image3_corners[i][0][0] = image3_corners[i][0][0] - min_x
        image3_corners[i][0][1] = image3_corners[i][0][1] - min_y
        image4_corners[i][0][0] = image4_corners[i][0][0] - min_x
        image4_corners[i][0][1] = image4_corners[i][0][1] - min_y

    print(image0_corners)
    print(image1_corners)
    print(image2_corners)
    print(image3_corners)
    print(image4_corners)

    max_height = 0
    max_width = 0

    for i in range(4):
        if image0_corners[i][0][1] > max_height:
            max_height = image0_corners[i][0][1]
        if image1_corners[i][0][1] > max_height:
            max_height = image1_corners[i][0][1]
        if image2_corners[i][0][1] > max_height:
            max_height = image2_corners[i][0][1]
        if image3_corners[i][0][1] > max_height:
            max_height = image3_corners[i][0][1]
        if image4_corners[i][0][1] > max_height:
            max_height = image4_corners[i][0][1]
        if image0_corners[i][0][0] > max_width:
            max_width = image0_corners[i][0][0]
        if image1_corners[i][0][0] > max_width:
            max_width = image1_corners[i][0][0]
        if image2_corners[i][0][0] > max_width:
            max_width = image2_corners[i][0][0]
        if image3_corners[i][0][0] > max_width:
            max_width = image3_corners[i][0][0]
        if image4_corners[i][0][0] > max_width:
            max_width = image4_corners[i][0][0]

    print("Max height: ", max_height)
    print("Max width: ", max_width)

    max_height = max_height
    max_width = max_width
    blank_image = np.zeros((int(max_height), int(max_width), 3), np.uint8)

    for i in range(4):
        cv2.circle(blank_image, (int(image0_corners[i][0][0]), int(image0_corners[i][0][1])), 50, (255, 0, 0), -1)

    for i in range(4):
        cv2.line(blank_image, (int(image0_corners[i][0][0]), int(image0_corners[i][0][1])), (int(image0_corners[(i+1)%4][0][0]), int(image0_corners[(i+1)%4][0][1])), (255, 0, 0), 15)


    for i in range(4):
        cv2.circle(blank_image, (int(image1_corners[i][0][0]), int(image1_corners[i][0][1])), 50, (0, 255, 0), -1)

    for i in range(4):
        cv2.line(blank_image, (int(image1_corners[i][0][0]), int(image1_corners[i][0][1])), (int(image1_corners[(i+1)%4][0][0]), int(image1_corners[(i+1)%4][0][1])), (0, 255, 0), 15)

    for i in range(4):
        cv2.circle(blank_image, (int(image2_corners[i][0][0]), int(image2_corners[i][0][1])), 50, (0, 0, 255), -1)

    for i in range(4):
        cv2.line(blank_image, (int(image2_corners[i][0][0]), int(image2_corners[i][0][1])), (int(image2_corners[(i+1)%4][0][0]), int(image2_corners[(i+1)%4][0][1])), (0, 0, 255), 15)

    for i in range(4):
        cv2.circle(blank_image, (int(image3_corners[i][0][0]), int(image3_corners[i][0][1])), 50, (255, 255, 0), -1)

    for i in range(4):
        cv2.line(blank_image, (int(image3_corners[i][0][0]), int(image3_corners[i][0][1])), (int(image3_corners[(i+1)%4][0][0]), int(image3_corners[(i+1)%4][0][1])), (255, 255, 0), 15)

    for i in range(4):
        cv2.circle(blank_image, (int(image4_corners[i][0][0]), int(image4_corners[i][0][1])), 50, (0, 255, 255), -1)

    for i in range(4):
        cv2.line(blank_image, (int(image4_corners[i][0][0]), int(image4_corners[i][0][1])), (int(image4_corners[(i+1)%4][0][0]), int(image4_corners[(i+1)%4][0][1])), (0, 255, 255), 15)



    # plt.figure()
    # plt.imshow(cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB))
    # plt.title('Corresponding corners between Transformed Images')
    # plt.show()


    if not os.path.exists('Corners/I4'):
        os.makedirs('Corners/I4')
    cv2.imwrite('Corners/I4/Corresponding_corners.jpg', blank_image)

    if not os.path.exists('Corners/I4'):
        os.makedirs('Corners/I4')
    cv2.imwrite('Corners/I4/Corresponding_corners.jpg', blank_image)

    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    X, Y = np.meshgrid(x, y)

    # Of the form x, y
    image0_mesh = np.dstack([X, Y])
    image0_mesh = image0_mesh.reshape(-1, 2)
    # Apply the homography matrix to the mesh of image 0
    image0_mesh_tr = apply_homography(image0_mesh.reshape(-1, 1, 2), homo_02)
    image0_mesh_tr = image0_mesh_tr.reshape(-1, 2)


    # Of the form x, y
    image1_mesh = np.dstack([X, Y])
    image1_mesh = image1_mesh.reshape(-1, 2)
    # Apply the homography matrix to the mesh of image 1
    image1_mesh_tr = apply_homography(image1_mesh.reshape(-1, 1, 2), homo_12)
    image1_mesh_tr = image1_mesh_tr.reshape(-1, 2)


    # Of the form x, y
    image2_mesh = np.dstack([X, Y])
    image2_mesh = image2_mesh.reshape(-1, 2)
    # Apply the homography matrix to the mesh of image 2
    image2_mesh_tr = image2_mesh
    image2_mesh_tr = image2_mesh_tr.reshape(-1, 2)


    # Of the form x, y
    image3_mesh = np.dstack([X, Y])
    image3_mesh = image3_mesh.reshape(-1, 2)
    # Apply the homography matrix to the mesh of image 3
    image3_mesh_tr = apply_homography(image3_mesh.reshape(-1, 1, 2), homo_32)
    image3_mesh_tr = image3_mesh_tr.reshape(-1, 2)


    # Of the form x, y
    image4_mesh = np.dstack([X, Y])
    image4_mesh = image4_mesh.reshape(-1, 2)
    # Apply the homography matrix to the mesh of image 4
    image4_mesh_tr = apply_homography(image4_mesh.reshape(-1, 1, 2), homo_42)
    image4_mesh_tr = image4_mesh_tr.reshape(-1, 2)



    filename0 = file_names[0]
    filename1 = file_names[1]
    filename2 = file_names[2]
    filename3 = file_names[3]
    filename4 = file_names[4]
    image0 = cv2.imread(path_to_img + filename0)    
    image1 = cv2.imread(path_to_img + filename1)
    image2 = cv2.imread(path_to_img + filename2)    
    image3 = cv2.imread(path_to_img + filename3)
    image4 = cv2.imread(path_to_img + filename4)    

    blank_image_0 = np.zeros((int(max_height), int(max_width), 3), np.uint8)
    blank_image_1 = np.zeros((int(max_height), int(max_width), 3), np.uint8)
    blank_image_2 = np.zeros((int(max_height), int(max_width), 3), np.uint8)
    blank_image_3 = np.zeros((int(max_height), int(max_width), 3), np.uint8)
    blank_image_4 = np.zeros((int(max_height), int(max_width), 3), np.uint8)
    result = np.zeros((int(max_height), int(max_width), 3), np.uint8)


    for i in tqdm(range(len(image0_mesh_tr))):
        x0 = int(image0_mesh[i][0])
        y0 = int(image0_mesh[i][1])
        x0_tr = int(image0_mesh_tr[i][0]-min_x)
        y0_tr = int(image0_mesh_tr[i][1]-min_y)
        try:
            if x0_tr < max_width and y0_tr < max_height and x0_tr >= 0 and y0_tr >= 0 and x0 < width and y0 < height and x0 >= 0 and y0 >= 0:
                blank_image_0[y0_tr, x0_tr] = image0[y0, x0]
                if x0_tr-1 >= 0 or x0_tr+1 < max_width or y0_tr-1 >= 0 or y0_tr+1 < max_height:
                    if blank_image_0[y0_tr, x0_tr-1, 0] == 0 and blank_image_0[y0_tr, x0_tr-1, 1] == 0 and blank_image_0[y0_tr, x0_tr-1, 2] == 0:
                        blank_image_0[y0_tr, x0_tr-1] = image0[y0, x0]
                    if blank_image_0[y0_tr, x0_tr+1, 0] == 0 and blank_image_0[y0_tr, x0_tr+1, 1] == 0 and blank_image_0[y0_tr, x0_tr+1, 2] == 0:
                        blank_image_0[y0_tr, x0_tr+1] = image0[y0, x0]
                    if blank_image_0[y0_tr-1, x0_tr, 0] == 0 and blank_image_0[y0_tr-1, x0_tr, 1] == 0 and blank_image_0[y0_tr-1, x0_tr, 2] == 0:
                        blank_image_0[y0_tr-1, x0_tr] = image0[y0, x0]
                    if blank_image_0[y0_tr+1, x0_tr, 0] == 0 and blank_image_0[y0_tr+1, x0_tr, 1] == 0 and blank_image_0[y0_tr+1, x0_tr, 2] == 0:
                        blank_image_0[y0_tr+1, x0_tr] = image0[y0, x0]
        except:
            pass

        try:
            x1 = int(image1_mesh[i][0])
            y1 = int(image1_mesh[i][1])
            x1_tr = int(image1_mesh_tr[i][0]-min_x)
            y1_tr = int(image1_mesh_tr[i][1]-min_y)
            if x1_tr < max_width and y1_tr < max_height and x1_tr >= 0 and y1_tr >= 0 and x1 < width and y1 < height and x1 >= 0 and y1 >= 0:
                blank_image_1[y1_tr, x1_tr] = image1[y1, x1]
                if x1_tr-1 >= 0 or x1_tr+1 < max_width or y1_tr-1 >= 0 or y1_tr+1 < max_height:
                    if blank_image_1[y1_tr, x1_tr-1, 0] == 0 and blank_image_1[y1_tr, x1_tr-1, 1] == 0 and blank_image_1[y1_tr, x1_tr-1, 2] == 0:
                        blank_image_1[y1_tr, x1_tr-1] = image1[y1, x1]
                    if blank_image_1[y1_tr, x1_tr+1, 0] == 0 and blank_image_1[y1_tr, x1_tr+1, 1] == 0 and blank_image_1[y1_tr, x1_tr+1, 2] == 0:
                        blank_image_1[y1_tr, x1_tr+1] = image1[y1, x1]
                    if blank_image_1[y1_tr-1, x1_tr, 0] == 0 and blank_image_1[y1_tr-1, x1_tr, 1] == 0 and blank_image_1[y1_tr-1, x1_tr, 2] == 0:
                        blank_image_1[y1_tr-1, x1_tr] = image1[y1, x1]
                    if blank_image_1[y1_tr+1, x1_tr, 0] == 0 and blank_image_1[y1_tr+1, x1_tr, 1] == 0 and blank_image_1[y1_tr+1, x1_tr, 2] == 0:
                        blank_image_1[y1_tr+1, x1_tr] = image1[y1, x1]
        except:
            pass

        try:
            x2 = int(image2_mesh[i][0])
            y2 = int(image2_mesh[i][1])
            x2_tr = int(image2_mesh_tr[i][0]-min_x)
            y2_tr = int(image2_mesh_tr[i][1]-min_y)
            if x2_tr < max_width and y2_tr < max_height and x2_tr >= 0 and y2_tr >= 0 and x2 < width and y2 < height and x2 >= 0 and y2 >= 0:
                blank_image_2[y2_tr, x2_tr] = image2[y2, x2]
                if x2_tr-1 >= 0 or x2_tr+1 < max_width or y2_tr-1 >= 0 or y2_tr+1 < max_height:
                    if blank_image_2[y2_tr, x2_tr-1, 0] == 0 and blank_image_2[y2_tr, x2_tr-1, 1] == 0 and blank_image_2[y2_tr, x2_tr-1, 2] == 0:
                        blank_image_2[y2_tr, x2_tr-1] = image2[y2, x2]
                    if blank_image_2[y2_tr, x2_tr+1, 0] == 0 and blank_image_2[y2_tr, x2_tr+1, 1] == 0 and blank_image_2[y2_tr, x2_tr+1, 2] == 0:
                        blank_image_2[y2_tr, x2_tr+1] = image2[y2, x2]
                    if blank_image_2[y2_tr-1, x2_tr, 0] == 0 and blank_image_2[y2_tr-1, x2_tr, 1] == 0 and blank_image_2[y2_tr-1, x2_tr, 2] == 0:
                        blank_image_2[y2_tr-1, x2_tr] = image2[y2, x2]
                    if blank_image_2[y2_tr+1, x2_tr, 0] == 0 and blank_image_2[y2_tr+1, x2_tr, 1] == 0 and blank_image_2[y2_tr+1, x2_tr, 2] == 0:
                        blank_image_2[y2_tr+1, x2_tr] = image2[y2, x2]
        except:
            pass

        try:
            x3 = int(image3_mesh[i][0])
            y3 = int(image3_mesh[i][1])
            x3_tr = int(image3_mesh_tr[i][0]-min_x)
            y3_tr = int(image3_mesh_tr[i][1]-min_y)
            if x3_tr < max_width and y3_tr < max_height and x3_tr >= 0 and y3_tr >= 0 and x3 < width and y3 < height and x3 >= 0 and y3 >= 0:
                blank_image_3[y3_tr, x3_tr] = image3[y3, x3]
                if x3_tr-1 >= 0 or x3_tr+1 < max_width or y3_tr-1 >= 0 or y3_tr+1 < max_height:
                    if blank_image_3[y3_tr, x3_tr-1, 0] == 0 and blank_image_3[y3_tr, x3_tr-1, 1] == 0 and blank_image_3[y3_tr, x3_tr-1, 2] == 0:
                        blank_image_3[y3_tr, x3_tr-1] = image3[y3, x3]
                    if blank_image_3[y3_tr, x3_tr+1, 0] == 0 and blank_image_3[y3_tr, x3_tr+1, 1] == 0 and blank_image_3[y3_tr, x3_tr+1, 2] == 0:
                        blank_image_3[y3_tr, x3_tr+1] = image3[y3, x3]
                    if blank_image_3[y3_tr-1, x3_tr, 0] == 0 and blank_image_3[y3_tr-1, x3_tr, 1] == 0 and blank_image_3[y3_tr-1, x3_tr, 2] == 0:
                        blank_image_3[y3_tr-1, x3_tr] = image3[y3, x3]
                    if blank_image_3[y3_tr+1, x3_tr, 0] == 0 and blank_image_3[y3_tr+1, x3_tr, 1] == 0 and blank_image_3[y3_tr+1, x3_tr, 2] == 0:
                        blank_image_3[y3_tr+1, x3_tr] = image3[y3, x3]
        except:
            pass

        try:
            x4 = int(image4_mesh[i][0])
            y4 = int(image4_mesh[i][1])
            x4_tr = int(image4_mesh_tr[i][0]-min_x)
            y4_tr = int(image4_mesh_tr[i][1]-min_y)
            if x4_tr < max_width and y4_tr < max_height and x4_tr >= 0 and y4_tr >= 0 and x4 < width and y4 < height and x4 >= 0 and y4 >= 0:
                blank_image_4[y4_tr, x4_tr] = image4[y4, x4]
                if x4_tr-1 >= 0 or x4_tr+1 < max_width or y4_tr-1 >= 0 or y4_tr+1 < max_height:
                    if blank_image_4[y4_tr, x4_tr-1, 0] == 0 and blank_image_4[y4_tr, x4_tr-1, 1] == 0 and blank_image_4[y4_tr, x4_tr-1, 2] == 0:
                        blank_image_4[y4_tr, x4_tr-1] = image4[y4, x4]
                    if blank_image_4[y4_tr, x4_tr+1, 0] == 0 and blank_image_4[y4_tr, x4_tr+1, 1] == 0 and blank_image_4[y4_tr, x4_tr+1, 2] == 0:
                        blank_image_4[y4_tr, x4_tr+1] = image4[y4, x4]
                    if blank_image_4[y4_tr-1, x4_tr, 0] == 0 and blank_image_4[y4_tr-1, x4_tr, 1] == 0 and blank_image_4[y4_tr-1, x4_tr, 2] == 0:
                        blank_image_4[y4_tr-1, x4_tr] = image4[y4, x4]
                    if blank_image_4[y4_tr+1, x4_tr, 0] == 0 and blank_image_4[y4_tr+1, x4_tr, 1] == 0 and blank_image_4[y4_tr+1, x4_tr, 2] == 0:
                        blank_image_4[y4_tr+1, x4_tr] = image4[y4, x4]
        except:
            pass

    if not os.path.exists('Transformed_Images/I4'):
        os.makedirs('Transformed_Images/I4')

    cv2.imwrite('Transformed_Images/I4/' + filename0, blank_image_0)
    cv2.imwrite('Transformed_Images/I4/' + filename1, blank_image_1)
    cv2.imwrite('Transformed_Images/I4/' + filename2, blank_image_2)
    cv2.imwrite('Transformed_Images/I4/' + filename3, blank_image_3)
    cv2.imwrite('Transformed_Images/I4/' + filename4, blank_image_4)

    if not os.path.exists('Homography_Matrices/I4'):
        os.makedirs('Homography_Matrices/I4')
    np.savetxt('Homography_Matrices/I4/homography_02.txt', homo_02)
    np.savetxt('Homography_Matrices/I4/homography_12.txt', homo_12)
    np.savetxt('Homography_Matrices/I4/homography_22.txt', homo_22)
    np.savetxt('Homography_Matrices/I4/homography_32.txt', homo_32)
    np.savetxt('Homography_Matrices/I4/homography_42.txt', homo_42)


    np.savetxt('Corners/I4/corners0.txt', transformed_corners_0.flatten())
    np.savetxt('Corners/I4/corners1.txt', transformed_corners_1.flatten())
    np.savetxt('Corners/I4/corners2.txt', image2_corners.flatten())
    np.savetxt('Corners/I4/corners3.txt', transformed_corners_3.flatten())
    np.savetxt('Corners/I4/corners4.txt', transformed_corners_4.flatten())

    img0 = cv2.imread('Transformed_Images/I4/' + filename0)
    img1 = cv2.imread('Transformed_Images/I4/' + filename1)
    img2 = cv2.imread('Transformed_Images/I4/' + filename2)
    img3 = cv2.imread('Transformed_Images/I4/' + filename3)
    img4 = cv2.imread('Transformed_Images/I4/' + filename4)


    image0 = blank_image_0
    image1 = blank_image_1
    image2 = blank_image_2
    image3 = blank_image_3
    image4 = blank_image_4

    result = image0
    mask = np.any(image1 != 0, axis=-1)
    result[mask] = image1[mask]
    mask = np.any(image2 != 0, axis=-1)
    result[mask] = image2[mask]
    mask = np.any(image3 != 0, axis=-1)
    result[mask] = image3[mask]
    mask = np.any(image4 != 0, axis=-1)
    result[mask] = image4[mask]

    # plt.figure()
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.title('Stitched Image')
    # plt.show()




    # plt.figure()
    smoothened_img = cv2.medianBlur(result, 25)
    # plt.imshow(smoothened_img)
    cv2.imwrite('Results/I4_filtered.jpg', smoothened_img)
    cv2.imwrite('Results/I4.jpg', result)



