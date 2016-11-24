import cv2
import matplotlib.pyplot as plt
import numpy as np

# -------------RESIZING IMAGE--------------------
## we need to keep in mind aspect ratio so the image does
## not look skewed or distorted -- therefore, we calculate
## the ratio of the new image to the old image
def resize_img(img):
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
     
    # perform the actual resizing of the image and show it
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def show_image(img, name=''):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def show_images(img_array, name=''):
    cv2.imshow(name, np.hstack(img_array))
    cv2.waitKey(0)
    
def show_img_plt(img):
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
# Histogram Equalization
def equalize_hist(img):
    return cv2.equalizeHist(img)

# Turn intensity that is greater than i to white, and leave the rest to black
# IMG in grayscale
def filter_intensity(img, intensity):
    return (img > intensity).astype(int)*255

def filter_rgb(img):
    # define the list of boundaries
    boundaries = [
        ([15, 50, 15], [125, 255, 125]) # Green
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
     
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(img, lower, upper)
            output = cv2.bitwise_and(img, resized, mask = mask)
     
            # show the images
            cv2.imshow("images", np.hstack([img, output]))
            cv2.waitKey(0)
            
def filter_hsv(img):
    # define the list of boundaries
    boundaries = [
        ([40, 0, 0], [45, 255, 255]) # Green
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
     
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(img, lower, upper)
            mask = 255 - mask
            output = cv2.bitwise_and(img, img, mask = mask)
           
            return output
        
def filter_hsv_in_range(img, lower_bound, upper_bound):
    # create NumPy arrays from the boundaries
    lower = np.array(lower_bound, dtype = "uint8")
    upper = np.array(upper_bound, dtype = "uint8")
                 
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask = mask)
                       
    return output

def calculate_intensities(img):
    intensity_value = 0
    for row_array in img:
        for pixel in row_array:
            intensity_value += pixel

    return intensity_value

def calculate_average_hsv(hsv_img):
    h_total = 0
    s_total = 0
    v_total = 0
    count = 0 
    for hsv_array in hsv_img:
        for hsv in hsv_array:
            h = hsv[0]
            s = hsv[1]
            v = hsv[2]
            if s > 0 and v > 0:
                h_total += h
                s_total += s
                v_total += v
                count += 1
    h_avg = h_total/count
    s_avg = s_total/count
    v_avg = v_total/count
    return h_avg, s_avg, v_avg

def hsv2gray(hsv_img):
    return cv2.cvtColor(
        cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL), cv2.COLOR_BGR2GRAY)

def load_hsv_image(file_name):
    img_bgr = resize_img(cv2.imread(file_name, 1))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def extract_initial_information(img):
    hsv_1 = filter_hsv(img)
    hsv_1_gray = hsv2gray(hsv_1)
    h1_avg, s1_avg, v1_avg = calculate_average_hsv(hsv_1)
    moment = cv2.moments(hsv_1_gray)
    
    result = {
        "h": h1_avg,
        "s": s1_avg,
        "v": v1_avg,
        "image": hsv_1
    }
    result.update(extract_mass_and_centroid_from_moments(moment))

    return result

def extract_mass_and_centroid_from_moments(moment):
    x = moment['m10']/moment['m00']
    y = moment['m01']/moment['m00']
    m = moment['m00']

    return  {
        "area": m,
        "x": x,
        "y": y
    }


def calculate_mass_and_centroid(img, lower_bound, upper_bound):
    image_filtered = filter_hsv_in_range(img, lower_bound, upper_bound)
    image_gray = hsv2gray(image_filtered)
    moment = cv2.moments(image_gray)

    result = {
        "image": image_filtered
    }
    result.update(extract_mass_and_centroid_from_moments(moment))
    return result


def find_hue_values():
    hue_count_dict = {}

    for pixel_array in initial_setup_info['image']:
        for pixel in pixel_array:
            h = pixel[0]
            if h > 0:
                if h not in hue_count_dict:
                    hue_count_dict[h] = 1
                else:
                    hue_count_dict[h] += 1

    max_y = max(hue_count_dict.values())
    hue_values = []

    for keys, value in hue_count_dict.iteritems():
        if value*1.0/max_y > 0.6:
            hue_values.append(keys)
