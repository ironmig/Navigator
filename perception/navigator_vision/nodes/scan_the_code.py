#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from mil_ros_tools import Image_Subscriber, Image_Publisher, rosmsg_to_numpy
from mil_vision_tools import auto_canny, RectFinder, ImageMux, contour_mask, putText_ul, roi_enclosing_points, rect_from_roi, Threshold
from collections import deque
from navigator_vision import ScanTheCodeClassifier
from mil_msgs.srv import ObjectDBQuery, ObjectDBQueryRequest
import tf
from image_geometry import PinholeCameraModel
from threading import Lock
from copy import copy
from std_msgs.msg import String


class ScanTheCodePerception(object):
    LED_HEIGHT = 0.38608
    LED_WIDTH = 0.19304
    ROI_MIN_Y = -0.7
    ROI_MAX_Y = 0


    def __init__(self):
        self.lock = Lock()
        self.enabled = False
        self.tf_listener = tf.TransformListener()
        self.debug = False
        self.roi = None
        self.img = None
        self.get_params()
        self.rect_finder = RectFinder(self.LED_HEIGHT, self.LED_WIDTH)
        self.db_service = rospy.ServiceProxy('/database/requests', ObjectDBQuery)
        self.sub = Image_Subscriber(self.image_topic, self.img_cb)
        info = self.sub.wait_for_camera_info()
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(info)
        self.update_roi()  # Update region of interest at start if possible
        self.debug_pub = Image_Publisher('~debug_image')
        res = 2
        self.image_mux = ImageMux(size=(info.height * res, info.width * res), shape=(2, 2),
                                  labels=['Original', 'Threshold', 'Edges', 'Classification'])
        self.solution_pub = rospy.Publisher('~solution', String, queue_size=1)
        self.classification_list = deque()  # "DECK - AH - WAY - WAY"
        self.enabled = True
        rospy.Timer(rospy.Duration(0.25), self.update_roi)  # Regularly update ROI

    def filter_stc_points(self, points):
        '''
        Filter out some of the 3D points in camera frame via heuristics such as a range of heights (y).
        For now does nothing because camera tf is off, making this impossible.
        '''
        return points

    def get_params(self):
        '''
        Set several constants used for image processing and classification
        from ROS params for runtime configurability.
        '''
        self.debug = rospy.get_param('~debug', True)
        self.image_topic = rospy.get_param('~image_topic', '/camera/seecam/image_rect_color')
        self.min_contour_area = rospy.get_param('~min_contour_area', 250)
        self.filter_d = rospy.get_param('filter_d', 5)
        self.filter_sigma = rospy.get_param('filter_sigma', 50)
        self.threshold = Threshold.from_dict(rospy.get_param('~threshold', {'LAB':[[150, 130, 100], [255, 140, 130]]}), in_space='BGR')
        self.roi_border = rospy.get_param('~roi_border', (20, 0))
        self.classifier = ScanTheCodeClassifier()
        self.classifier.train_from_csv()

    def update_roi(self, timer_obj=None):
        '''
        Update the region of interest where the LED panel os
        '''
        # Get points from database
        try:
            res = self.db_service(ObjectDBQueryRequest(name='stc'))
        except rospy.ServiceException as e:
            rospy.logwarn('Database service error: {}'.format(e))
            return
        if not res.found:
            rospy.logwarn('Scan the code object not found')
            return
        obj = res.objects[0]

        # Get transform
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.camera_model.tfFrame(),
                                                            obj.header.frame_id, rospy.Time(0))
        except tf.Exception as e:
            rospy.logwarn('TF error betwen {} and {}: {}'.format(self.camera_model.tfFrame(), obj.header.frame_id, e))
            return
        P = np.array(trans)
        R = tf.transformations.quaternion_matrix(rot)[:3, :3]
        points = rosmsg_to_numpy(obj.points)
        points_transformed = P + (R.dot(points.T)).T
        points_transformed = self.filter_stc_points(points_transformed)

        # Set ROI
        self.lock.acquire()
        self.roi = roi_enclosing_points(self.camera_model, points_transformed, border=self.roi_border)
        self.lock.release()

    def get_stc_mask(self, img):
        '''
        Return the contour corosponding to the LED panel of scan the code in the image, if any contours
        meet the criteria. The algorithim does the following:
        1) Convert to LAB and blur (gaussian)
        2) Loosely threshold in LAB colorspace to making edge detection easier
        3) Detect edges with Canny
        4) Find contours in edge image
        5) Filter out contours with below a fixed area threshold
        6) Sort remaining contours by their match to the model of the rectangle
        7) Traverse sorted list, finding first one which can be approximated as a rectangle
        '''
        blur = cv2.GaussianBlur(img, (11, 11), 20)
        threshold = self.threshold(blur)
        edges = auto_canny(threshold)
        if self.debug:
            self.image_mux[1] = threshold
            self.image_mux[2] = edges
        c, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        filtered_contours = filter(lambda c: cv2.contourArea(c) > self.min_contour_area, contours)
        if len(filtered_contours) == 0:
            rospy.logdebug('No contours meeting area minimum')
            return None

        matches = map(self.rect_finder.verify_contour, filtered_contours)
        matches_sorted_index = np.argsort(matches)
        sorted_matches = np.array(filtered_contours)[matches_sorted_index]
        rospy.loginfo('best {}'.format(matches[matches_sorted_index[0]]))
        rospy.loginfo('worse {}'.format(matches[matches_sorted_index[-1]]))
        match = None
        for i in sorted_matches:
            pts = self.rect_finder.get_corners(i, epsilon_range=(0.01, 0.05))
            if pts is None:
                continue
            _, _, width, height = cv2.boundingRect(pts)
            # STC is always upright
            ratio = float(height) / width
            # print 'ratio ', ratio
            if ratio < 1 or ratio > 5:
                continue
            match = i
            break
        if match is None:
            rospy.logdebug('No contours which can be approximated as rectangle')
            return
        return match

    def detect_pattern(self):
        '''
        Looks at most recent 5 detected colors to find a 3 color pattern solution.
        If the list is in the form ['o', 'x', 'x', 'x', 'o'], with 3 colors ('x') surrounded
        on either side by an off ('o'), return those 3 colors in the middle as a string like 'rgb'.
        Otherwise, return None
        '''
        if len(self.classification_list) != 5:
            return None
        if self.classification_list[0] == 'o' and self.classification_list[4] == 'o'\
           and self.classification_list[1] != 'o' and self.classification_list[2] != 'o'\
           and self.classification_list[3] != 'o':
            solution = self.classification_list[1] + self.classification_list[2] + self.classification_list[3]
            self.solution_pub.publish(solution)
            rospy.loginfo('SOLUTIONS ' + solution)
            return solution
        return None

    def classify_panel(self, img, contour):
        '''
        Classify the color (or off) of the LED panel and update the buffer
        of recent classifications.
        '''
        # Get classification index and string from classifier
        mask = contour_mask(contour, img_shape=img.shape)
        prediction = self.classifier.classify(img, mask)[0]  # class index, ex: 0
        label = self.classifier.CLASSES[prediction]  # class string, ex: 'stc_red'
        symbol = label[4]  # one character representation of label ex: 'r'

        # update buffer of recent colors if it has changed
        if len(self.classification_list) == 0 or self.classification_list[-1] != symbol:
            if len(self.classification_list) >= 5:  # Only keep most recent 5 colors (needed to see pattern)
                self.classification_list.popleft()
            self.classification_list.append(symbol)
            self.detect_pattern()
        rospy.loginfo('FOUND: ' + ''.join(self.classification_list))
        if self.debug:
            debug = cv2.bitwise_or(img, img, mask=mask)
            putText_ul(debug, label, (0, 0), fontScale=1, thickness=2)
            self.image_mux[3] = debug

    def img_cb(self, img):
        if not self.enabled:  # Skip if not initialized/enabled
            return

        # Copy ROI locally
        self.lock.acquire()
        roi = copy(self.roi)
        self.lock.release()
        if roi is None:  # Skip if ROI not found yet
            rospy.logwarn_throttle(1.0, 'no roi')
            return

        # If at least one subscribe to debug image, enable producing of debug images
        self.debug = self.debug_pub.im_pub.get_num_connections() > 0

        # Crop image using ROI from object database
        cropped = img[roi]
        if self.debug:
            self.image_mux[0] = cropped

        # Get contour of LED panel from
        match = self.get_stc_mask(cropped)
        if match is not None: # Classify panel and look for pattern if match found
            self.classify_panel(cropped, match)

        # Publish debug image
        if self.debug:
            self.debug_pub.publish(self.image_mux())


if __name__ == '__main__':
    rospy.init_node('scan_the_code_perception')
    s = ScanTheCodePerception()
    rospy.spin()
