import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

all_image = io.imread('../data/vhr_data/vhr_test.tif')
result_image = io.imread('./model/result/image_predict_fusion.tif') + 1

image_w = all_image.shape[0]
image_h = all_image.shape[1]

w_size = 300
h_size = 400
w_range = int(np.ceil(image_w/w_size))
h_range = int(np.ceil(image_h/h_size))
save_image = np.zeros([image_w,image_h,3])
save_image_2 = np.zeros([image_w,image_h])
rgb_image = np.zeros([image_w,image_h])

positive_cate = [10]
negative_cate = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20]

for i in range(len(negative_cate)):
    cate = negative_cate[i]
    index =np.where(result_image==cate)
    result_image[index] = 0

for i in range(len(positive_cate)):
    cate = positive_cate[i]
    index =np.where(result_image==cate)
    result_image[index] = 1

for w_i in tqdm(range(w_range)):
    for h_j in range(h_range):

        x00 = w_i*w_size
        x11 = (w_i+1)*w_size
        if x11>image_w:
            x11 = image_w
        y00 = h_j*h_size
        y11 = (h_j+1)*h_size
        if y11>image_h:
            y11 = image_h

        img = all_image[x00:x11,y00:y11,:]
        result_img = result_image[x00:x11,y00:y11]
        color_img = rgb_image[x00:x11,y00:y11]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img,(3,3),0)
        canny = cv2.Canny(img, 50, 200)
        cv2.imshow('Canny', canny)

        lines = cv2.HoughLines(canny,1,np.pi/180,160)
        if lines == None:
            print('ok!')

        else :
            if len(lines)<8:
                print('ok!')
            else:
                lines1 = lines[:,0,:]
                rhos_all = lines1[:,0]
                theta_all = lines1[:,1]
                index = np.argsort(rhos_all)  # index
                sort_rho_init = lines1[np.argsort(rhos_all)]
                num_delet = rhos_all[np.argsort(rhos_all)]
                delet_idx = []
                sort_rho = sort_rho_init
                idx = 0
                for i in range(len(sort_rho_init)-1):
                    if num_delet[i+1]-num_delet[i]<5:
                        sort_rho = np.delete(sort_rho, i+1-idx, 0)
                        idx = idx+1

                for line_single in range(len(sort_rho)-1):
                    rho,theta = sort_rho[line_single]
                    rho2,theta2 = sort_rho[line_single+1]
                    center_puixel = []
                    center_all = []
                    for img_i in range(img.shape[0]):
                        for img_j in range(img.shape[1]):
                            x0 = (rho-img_j*np.cos(theta))*1.0/np.sin(theta)
                            x1 = (rho2-img_j*np.cos(theta2))*1.0/np.sin(theta2)
                            if x0<img_i and x1>img_i:
                                center_puixel.append(result_img[img_i,img_j])
                                center_all.append([img_i,img_j])
                    center_num = sum(center_puixel)
                    if len(center_puixel)==0:
                        continue
                    if center_num*1.0/len(center_puixel)>0.9:
                        road_maxsize = rho2-rho
                        theta_2 = theta2-theta
                        if road_maxsize>25 and theta_2<0.8:
                            center_01 = np.array(center_all)
                            center_0 = center_01[:,0]
                            print(np.max(center_0))
                            print(np.min(center_0))
                            for color in range(len(center_01)):
                                position1 = center_01[color,0]
                                position2 = center_01[color,1]
                                color_img[position1,position2] = 1

                    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
                        pt1 = (int(rho / np.cos(theta)), 0)
                        pt2 = (int((rho - img.shape[0] * np.sin(theta)) / np.cos(theta)), img.shape[0])
                        cv2.line(img,pt1,pt2,(255,0,0),1)
                    else:
                        pt1 = (0, int(rho / np.sin(theta)))
                        pt2 = (img.shape[1], int((rho - img.shape[1] * np.cos(theta)) / np.sin(theta)))
                        cv2.line(img, pt1, pt2, (255,0,0), 1)

        save_image_2[x00:x11,y00:y11] = color_img
        save_image[x00:x11,y00:y11,:] = img

save_image = save_image.astype(np.uint8)
io.imsave('./Canny_roads.tif', save_image)
io.imsave('./Canny_result.tif', save_image_2)
