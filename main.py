import cv2
import matplotlib.pyplot as plt
import numpy as np
from helper import *

# -------------READING IMAGE---------------------
benchmark = load_hsv_image('DSC_0021.jpg')

center = load_hsv_image('DSC_0026.jpg')
right = load_hsv_image('DSC_0027.jpg')
left = load_hsv_image('DSC_0029.jpg')

multi_people = load_hsv_image('DSC_0030.jpg')
overlap = load_hsv_image('DSC_0031.jpg')
#test_image_4 = load_hsv_image('paint_test_image_smaller.jpg')


# ----------------Filtering Background--------------------
initial_setup_info = extract_initial_information(benchmark)

#--------------------Identifyng objects---------------------
lower_bound = [initial_setup_info['h']-10, initial_setup_info['s']-20, initial_setup_info['v']-20]
upper_bound = [initial_setup_info['h']+50, 255, 255]

info_1 = calculate_mass_and_centroid(center, lower_bound, upper_bound)
info_2 = calculate_mass_and_centroid(right, lower_bound, upper_bound)
info_3 = calculate_mass_and_centroid(left, lower_bound, upper_bound)
info_4= calculate_mass_and_centroid(multi_people, lower_bound, upper_bound)
info_5 = calculate_mass_and_centroid(overlap, lower_bound, upper_bound)
    
print 'center'
print info_1['area'], info_1['x'], info_1['y']

print 'right'
print info_2['area'], info_2['x'], info_2['y']

print 'left'
print info_3['area'], info_3['x'], info_3['y']

print 'multi_people'
print info_4['area'], info_4['x'], info_4['y']

print 'overlap'
print info_5['area'], info_5['x'], info_5['y']

show_images([info_1['image'], info_2['image'], info_3['image']])
#show_images([info_4['image'], info_5['image']])
