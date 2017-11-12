#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from mil_ros_tools import Image_Subscriber, Image_Publisher
from mil_vision_tools import auto_canny, RectFinder, ImageMux
import threading
import pandas
from sklearn.naive_bayes import GaussianNB


class ScanTheCodePerception(object):
    LED_HEIGHT = 0.38608
    LED_WIDTH = 0.19304
    CLASSES = ['off', 'red', 'green', 'blue', 'yellow']

    def __init__(self):
        self.get_params()
        self.rect_finder = RectFinder(self.LED_HEIGHT, self.LED_WIDTH)
        self.sub = Image_Subscriber(self.image_topic, self.img_cb)
        info = self.sub.wait_for_camera_info()
        if self.debug:
            self.debug_pub = Image_Publisher('~debug_image')
            res = 2
            self.image_mux = ImageMux(size=(info.height*res, info.width*res), shape=(2,2),
                                      labels=['Original', 'Contours', '', ''])

    def get_features(self, image, mask):
        features = np.zeros(9)
        mean = cv2.mean(image, mask)
        mean = np.array([[mean[:3]]], dtype=np.uint8)
        mean_hsv = cv2.cvtColor(mean, cv2.COLOR_BGR2HSV)
        mean_lab = cv2.cvtColor(mean, cv2.COLOR_BGR2LAB)
        features[0:3] = mean.flatten()
        features[3:6] = mean_hsv.flatten()
        features[6:] = mean_lab.flatten()
        return features

    def get_params(self):
        self.debug = rospy.get_param('~debug', True)
        self.image_topic = rospy.get_param('~image_topic', '/stereo/right/image_rect_color')
        self.min_contour_area = rospy.get_param('~min_contour_area', 250)
        self.color_calibration_file = rospy.get_param('~calibration', None)
        if self.color_calibration_file is None:
            import rospkg
            import os
            rospack = rospkg.RosPack()
            nlaunch = rospack.get_path('navigator_launch')
            self.color_calibration_file = os.path.join(nlaunch, 'config', 'stc_colors.csv')
        df = pandas.read_csv(self.color_calibration_file, index_col=0)
        values = np.array(df.values, dtype=int) # Convert to int
        np.random.shuffle(values) # Shuffle so training / testing are evenly distributed about bags

        # Do a quick do it work test
        split = int(0.75 * values.shape[0])
        train = values[:split, :]
        test = values[split:, :]
        clf  = GaussianNB()
        clf.fit(train[:, 1:], train[:, 0])
        rospy.loginfo('Color calibration has {}% accuracy on validation set'.format(clf.score(test[:, 1:], test[:, 0])*100))

        self.color_classifier = GaussianNB()
        self.color_classifier.fit(values[:, 1:], values[:, 0])


    def img_cb(self, img):
        edges = auto_canny(img)
        c, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        filtered_contours = filter(lambda c : cv2.contourArea(c) > self.min_contour_area, contours)
        if len(filtered_contours) == 0:
            print 'none after area fiter'
            return

        matches = map(self.rect_finder.verify_contour, filtered_contours)
        matches_sorted_index = np.argsort(matches)
        sorted_contours = np.array(filtered_contours)[matches_sorted_index]

        match = None
        for cnt in sorted_contours:
            pts = self.rect_finder.get_corners(cnt, epsilon_range=(0.01, 0.05))
            if pts is None:
                continue
            _, _, width, height = bounding_rect = cv2.boundingRect(pts)
            # STC is always upright
            ratio = float(height) / width
            print 'ratio ', ratio
            if ratio < 1 or ratio > 5:
                continue
            print 'match'
            match = cnt
            break

        debug = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if match is not None:
            cv2.drawContours(debug, [match], -1, 255, -1)
            features = self.get_features(img, debug)
            prediction = self.color_classifier.predict(features)
            label = self.CLASSES[prediction]
            print label
            scale = 3
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, scale, thickness)
            cv2.putText(debug, label, (0, text_height), cv2.FONT_HERSHEY_COMPLEX_SMALL, scale, (255, 255, 255), thickness)
        debug = cv2.bitwise_or(img, img, mask=debug)

        if self.debug:
            self.image_mux[0] = img
            self.image_mux[1] = c
            self.image_mux[2] = debug
            self.debug_pub.publish(self.image_mux())

if __name__ == '__main__':
    rospy.init_node('scan_the_code_perception')
    s = ScanTheCodePerception()
    rospy.spin()
