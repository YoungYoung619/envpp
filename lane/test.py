#-*-coding:utf-8-*-
import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from collections import deque
##########################################################################
###### https://blog.csdn.net/baidu_39415947/article/details/80367630 #####
##########################################################################

def display(img,title,color=1):
    """ display image
    Args:
        img: rgb or grayscale
        title:figure title
        color:show image in color(1) or grayscale(0)
    """
    if color:
        plt.imshow(img)
    else:
        plt.imshow(img,cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def directional_gradient_binary(img,direction='x',thresh=[0,255]):
    """ use the sobel to calculate the gradient form img
    Args:
        img:Grayscale
        direction:x or y axis
        thresh:apply threshold on pixel intensity of gradient image
    Return:
        a binary image
    """
    if direction=='x':
        sobel = cv2.Sobel(img,cv2.CV_64F,1,0)
    elif direction=='y':
        sobel = cv2.Sobel(img,cv2.CV_64F,0,1)
    elif direction=='xy':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1)
    else:
        raise ValueError('Parameter direction wrong')
    sobel_abs = np.absolute(sobel)#absolute value
    scaled_sobel = np.uint8(sobel_abs*255/np.max(sobel_abs))
    binary_output = np.zeros_like(sobel)
    f = (scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])
    binary_output[f] = 1
    return binary_output


def color_binary(img,dst_format='HLS',channel=2,ch_thresh=[0,255]):
    """ Color thresholding on specified channel
    Args:
        img:RGB
        dst_format:destination format(HLS or HSV)
        channel: the retained channel, for lane detection, we need retain S channel,
                 for HLS format, set channel as 2; for HSV, set channel as 1
        ch_thresh:pixel intensity threshold on channel ch
    Return:
        a binary image
    """
    if dst_format =='HSV':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        ch_binary = np.zeros_like(img[:,:,int(channel)])
        ch_binary[(img[:,:,int(channel)]>=ch_thresh[0])&(img[:,:,int(channel)]<=ch_thresh[1])] = 1
    else:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        ch_binary = np.zeros_like(img[:,:,int(channel)])
        ch_binary[(img[:,:,int(channel)]>=ch_thresh[0])&(img[:,:,int(channel)]<=ch_thresh[1])] = 1
    return ch_binary


def birdView(img,M):
    """ Transform image to birdeye view
    Args:
        img:binary image
        M:transformation matrix
    Return:
         a wraped image
    """
    img_sz = (img.shape[1],img.shape[0])
    img_warped = cv2.warpPerspective(img,M,img_sz,flags = cv2.INTER_LINEAR)
    return img_warped


def perspective_transform(src_pts,dst_pts):
    """ perspective transform
    Args:
        source and destiantion points
    Return:
        M and Minv
    """
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts,src_pts)
    return {'M':M,'Minv':Minv}


def find_centroid(image,peak_thresh,window,showMe):
    '''
    find centroid in a window using histogram of hotpixels
    img:binary image
    window with specs {'x0','y0','width','height'}
    (x0,y0) coordinates of bottom-left corner of window
    return x-position of centroid ,peak intensity and hotpixels_cnt in window
    '''
    #crop image to window dimension
    mask_window = image[round(window['y0']-window['height']):round(window['y0']),
                        round(window['x0']):round(window['x0']+window['width'])]
    histogram = np.sum(mask_window,axis=0)
    centroid = np.argmax(histogram)
    hotpixels_cnt = np.sum(histogram)
    peak_intensity = histogram[centroid]
    if peak_intensity<=peak_thresh:
        centroid = int(round(window['x0']+window['width']/2))
        peak_intensity = 0
    else:
        centroid = int(round(centroid+window['x0']))
    if showMe:
        plt.plot(histogram)
        plt.title('Histogram')
        plt.xlabel('horzontal position')
        plt.ylabel('hot pixels count')
        plt.show()
    return (centroid,peak_intensity,hotpixels_cnt)


def find_starter_centroids(image,x0,peak_thresh,showMe):
    '''
    find starter centroids using histogram
    peak_thresh:if peak intensity is below a threshold use histogram on the full height of the image
    returns x-position of centroid and peak intensity
    '''
    window = {'x0':x0,'y0':image.shape[0],'width':image.shape[1]/2,'height':image.shape[0]/2}
    # get centroid
    centroid , peak_intensity,_ = find_centroid(image,peak_thresh,window,showMe)
    if peak_intensity<peak_thresh:
        window['height'] = image.shape[0]
        centroid,peak_intensity,_ = find_centroid(image,peak_thresh,window,showMe)
    return {'centroid':centroid,'intensity':peak_intensity}


def run_sliding_window(image, centroid_starter, sliding_window_specs, showMe=1):
    '''
    Run sliding window from bottom to top of the image and return indexes of the hotpixels associated with lane
    image:binary image
    centroid_starter:centroid starting location sliding window
    sliding_window_specs:['width','n_steps']
        width of sliding window
        number of steps of sliding window alog vertical axis
    return {'x':[],'y':[]}
        coordiantes of all hotpixels detected by sliding window
        coordinates of alll centroids recorded but not used yet!
    '''
    # Initialize sliding window
    window = {'x0': centroid_starter - int(sliding_window_specs['width'] / 2),
              'y0': image.shape[0], 'width': sliding_window_specs['width'],
              'height': round(image.shape[0] / sliding_window_specs['n_steps'])}
    hotpixels_log = {'x': [], 'y': []}
    centroids_log = []
    if showMe:
        out_img = (np.dstack((image, image, image)) * 255).astype('uint8')
    for step in range(sliding_window_specs['n_steps']):
        if window['x0'] < 0: window['x0'] = 0
        if (window['x0'] + sliding_window_specs['width']) > image.shape[1]:
            window['x0'] = image.shape[1] - sliding_window_specs['width']
        centroid, peak_intensity, hotpixels_cnt = find_centroid(image, peak_thresh, window, showMe=0)

        if hotpixels_cnt / (window['width'] * window['height']) > 0.6:
            window['width'] = window['width'] * 2
            window['x0'] = round(window['x0'] - window['width'] / 2)
            if (window['x0'] < 0): window['x0'] = 0
            if (window['x0'] + window['width']) > image.shape[1]:
                window['x0'] = image.shape[1] - window['width']
            centroid, peak_intensity, hotpixels_cnt = find_centroid(image, peak_thresh, window, showMe=0)

        mask_window = np.zeros_like(image)
        mask_window[window['y0'] - window['height']:window['y0'],
        window['x0']:window['x0'] + window['width']] \
            = image[window['y0'] - window['height']:window['y0'],
              window['x0']:window['x0'] + window['width']]

        hotpixels = np.nonzero(mask_window)
        # print(hotpixels_log['x'])

        hotpixels_log['x'].extend(hotpixels[0].tolist())
        hotpixels_log['y'].extend(hotpixels[1].tolist())
        # update record of centroid
        centroids_log.append(centroid)

        if showMe:
            cv2.rectangle(out_img,
                          (window['x0'], window['y0'] - window['height']),
                          (window['x0'] + window['width'], window['y0']), (0, 255, 0), 2)

            if int(window['y0']) == 72:
                plt.imshow(out_img)
                plt.show()
            '''
            print(window['y0'])
            plt.imshow(out_img)
            '''
        # set next position of window and use standard sliding window width
        window['width'] = sliding_window_specs['width']
        window['x0'] = round(centroid - window['width'] / 2)
        window['y0'] = window['y0'] - window['height']
    return hotpixels_log


def MahalanobisDist(x, y):
    """ Mahalanobis Distance for bi-variate distribution
    """
    covariance_xy = np.cov(x, y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x), np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]), inv_covariance_xy), diff_xy[i])))
    return md


def MD_removeOutliers(x, y, MD_thresh):
    """remove pixels outliers using Mahalonobis distance
    """
    MD = MahalanobisDist(x, y)
    threshold = np.mean(MD) * MD_thresh
    nx, ny, outliers = [], [], []
    for i in range(len(MD)):
        if MD[i] <= threshold:
            nx.append(x[i])
            ny.append(y[i])
        else:
            outliers.append(i)
    return (nx, ny)


def update_tracker(tracker,new_value):
    '''
    update tracker(self.bestfit or self.bestfit_real or radO Curv or hotpixels) with new coeffs
    new_coeffs is of the form {'a2':[val2,...],'a1':[va'1,...],'a0':[val0,...]}
    tracker is of the form {'a2':[val2,...]}
    update tracker of radius of curvature
    update allx and ally with hotpixels coordinates from last sliding window
    '''
    if tracker =='bestfit':
        bestfit['a0'].append(new_value['a0'])
        bestfit['a1'].append(new_value['a1'])
        bestfit['a2'].append(new_value['a2'])
    elif tracker == 'bestfit_real':
        bestfit_real['a0'].append(new_value['a0'])
        bestfit_real['a1'].append(new_value['a1'])
        bestfit_real['a2'].append(new_value['a2'])
    elif tracker == 'radOfCurvature':
        radOfCurv_tracker.append(new_value)
    elif tracker == 'hotpixels':
        allx.append(new_value['x'])
        ally.append(new_value['y'])


if __name__ == '__main__':
    path = "./data/test/"
    path = os.path.join(path, "*.jpg")
    img_names = glob.glob(path)


    original_img = cv2.imread(img_names[11])
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    #display(gray_img,"gray img",color=0)

    grad_x_b = directional_gradient_binary(gray_img, direction='x', thresh=[10, 255])
    grad_y_b = directional_gradient_binary(gray_img, direction='y', thresh=[10, 255])
    grad_xy_b = directional_gradient_binary(gray_img, direction='xy', thresh=[10, 255])
    grad_b = np.zeros_like(grad_x_b)
    grad_b[((grad_x_b == 1) & (grad_y_b == 1) & (grad_xy_b == 1))] = 1
    # display(grad_x_b, 'Gradient x', color=0)
    # display(grad_y_b, 'Gradient y', color=0)
    # display(grad_xy_b, 'Gradient xy', color=0)
    #display(grad_b, 'combine gradient', color=0)

    color_b = color_binary(rgb_img, dst_format='HLS',
                                    channel=2, ch_thresh=[60, 255])
    display(color_b, "color binary", color=0)

    ## combine the grad_x_b and color_b ##
    combined_output = np.zeros_like(grad_x_b)
    combined_output[((grad_x_b == 1) | (color_b == 1))] = 1
    display(combined_output, 'Combined output', color=0)

    ## ROI mask to filter some background ##
    mask = np.zeros_like(combined_output)
    vertices = np.array([[(200, 330), (1000, 330), (1280, 720), (0, 720)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 1)
    masked_image = cv2.bitwise_and(combined_output, mask)
    #display(masked_image, 'Masked', color=0)

    ## clean some noise ##
    min_sz = 200
    cleaned = morphology.remove_small_objects(masked_image.astype('bool'), min_size=min_sz, connectivity=2)
    #display(cleaned, 'cleaned', color=0)

    # original image to bird view (transformation)
    src_pts = np.float32([[173, 711], [480, 419], [604, 419], [979, 715]])
    dst_pts = np.float32([[80,720],[80,0],[1200,0],[1200,720]])
    transform_matrix = perspective_transform(src_pts, dst_pts)
    warped_image = birdView(cleaned * 1.0, transform_matrix['M'])
    #display(warped_image, 'BirdViews', color=0)

    # fine the start point to start slide windows#
    peak_thresh = 10
    showMe = 1
    centroid_starter_right = find_starter_centroids(warped_image, x0=warped_image.shape[1] / 2,
                                                    peak_thresh=peak_thresh, showMe=showMe)
    centroid_starter_left = find_starter_centroids(warped_image, x0=0, peak_thresh=peak_thresh,
                                                   showMe=showMe)

    ##
    showMe = 1
    sliding_window_specs = {'width': 120, 'n_steps': 10}
    log_lineLeft = run_sliding_window(warped_image, centroid_starter_left['centroid'], sliding_window_specs,
                                      showMe=showMe)
    log_lineRight = run_sliding_window(warped_image, centroid_starter_right['centroid'], sliding_window_specs,
                                       showMe=showMe)

    ###
    MD_thresh = 1.8
    log_lineLeft['x'], log_lineLeft['y'] = \
        MD_removeOutliers(log_lineLeft['x'], log_lineLeft['y'], MD_thresh)
    log_lineRight['x'], log_lineRight['y'] = \
        MD_removeOutliers(log_lineRight['x'], log_lineRight['y'], MD_thresh)

    ##
    buffer_sz = 5
    allx = deque([], maxlen=buffer_sz)
    ally = deque([], maxlen=buffer_sz)
    bestfit = {'a0': deque([], maxlen=buffer_sz),
               'a1': deque([], maxlen=buffer_sz),
               'a2': deque([], maxlen=buffer_sz)}
    bestfit_real = {'a0': deque([], maxlen=buffer_sz),
                    'a1': deque([], maxlen=buffer_sz),
                    'a2': deque([], maxlen=buffer_sz)}
    radOfCurv_tracker = deque([], maxlen=buffer_sz)

    update_tracker('hotpixels', log_lineRight)
    update_tracker('hotpixels', log_lineLeft)
    multiframe_r = {'x': [val for sublist in allx for val in sublist],
                    'y': [val for sublist in ally for val in sublist]}
    multiframe_l = {'x': [val for sublist in allx for val in sublist],
                    'y': [val for sublist in ally for val in sublist]}


    # fit to polynomial in pixel space
    def polynomial_fit(data):
        '''
        多项式拟合
        a0+a1 x+a2 x**2
        data:dictionary with x and y values{'x':[],'y':[]}
        '''
        a2, a1, a0 = np.polyfit(data['x'], data['y'], 2)
        return {'a0': a0, 'a1': a1, 'a2': a2}


    # merters per pixel in y or x dimension
    ym_per_pix = 12 / 450
    xm_per_pix = 3.7 / 911
    fit_lineLeft = polynomial_fit(multiframe_l)
    fit_lineLeft_real = polynomial_fit({'x': [i * ym_per_pix for i in multiframe_l['x']],
                                        'y': [i * xm_per_pix for i in multiframe_l['y']]})
    fit_lineRight = polynomial_fit(multiframe_r)
    fit_lineRight_real = polynomial_fit({'x': [i * ym_per_pix for i in multiframe_r['x']],
                                         'y': [i * xm_per_pix for i in multiframe_r['y']]})


    def predict_line(x0, xmax, coeffs):
        '''
        predict road line using polyfit cofficient
        x vaues are in range (x0,xmax)
        polyfit coeffs:{'a2':,'a1':,'a2':}
        returns array of [x,y] predicted points ,x along image vertical / y along image horizontal direction
        '''
        x_pts = np.linspace(x0, xmax - 1, num=xmax)
        pred = coeffs['a2'] * x_pts ** 2 + coeffs['a1'] * x_pts + coeffs['a0']
        return np.column_stack((x_pts, pred))


    fit_lineRight_singleframe = polynomial_fit(log_lineRight)
    fit_lineLeft_singleframe = polynomial_fit(log_lineLeft)
    var_pts = np.linspace(0, rgb_img.shape[0] - 1, num=rgb_img.shape[0])
    pred_lineLeft_singleframe = predict_line(0, rgb_img.shape[0], fit_lineLeft_singleframe)
    pred_lineRight_sigleframe = predict_line(0, rgb_img.shape[0], fit_lineRight_singleframe)
    plt.plot(pred_lineLeft_singleframe[:, 1], pred_lineLeft_singleframe[:, 0], 'b-', label='singleframe', linewidth=1)
    plt.plot(pred_lineRight_sigleframe[:, 1], pred_lineRight_sigleframe[:, 0], 'b-', label='singleframe', linewidth=1)
    plt.show()


    def compute_radOfCurvature(coeffs, pt):
        return ((1 + (2 * coeffs['a2'] * pt + coeffs['a1']) ** 2) ** 1.5) / np.absolute(2 * coeffs['a2'])


    pt_curvature = rgb_img.shape[0]
    radOfCurv_r = compute_radOfCurvature(fit_lineRight_real, pt_curvature * ym_per_pix)
    radOfCurv_l = compute_radOfCurvature(fit_lineLeft_real, pt_curvature * ym_per_pix)
    average_radCurv = (radOfCurv_r + radOfCurv_l) / 2

    center_of_lane = (pred_lineLeft_singleframe[:, 1][-1] + pred_lineRight_sigleframe[:, 1][-1]) / 2
    offset = (rgb_img.shape[1] / 2 - center_of_lane) * xm_per_pix

    side_pos = 'right'
    if offset < 0:
        side_pos = 'left'
    wrap_zero = np.zeros_like(gray_img).astype(np.uint8)
    color_wrap = np.dstack((wrap_zero, wrap_zero, wrap_zero))
    left_fitx = fit_lineLeft_singleframe['a2'] * var_pts ** 2 + fit_lineLeft_singleframe['a1'] * var_pts + \
                fit_lineLeft_singleframe['a0']
    right_fitx = fit_lineRight_singleframe['a2'] * var_pts ** 2 + fit_lineRight_singleframe['a1'] * var_pts + \
                 fit_lineRight_singleframe['a0']
    pts_left = np.array([np.transpose(np.vstack([left_fitx, var_pts]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, var_pts])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_wrap, np.int_([pts]), (0, 255, 0))
    cv2.putText(color_wrap, '|', (int(rgb_img.shape[1] / 2), rgb_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 8)
    cv2.putText(color_wrap, '|', (int(center_of_lane), rgb_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 8)
    newwrap = cv2.warpPerspective(color_wrap, transform_matrix['Minv'], (rgb_img.shape[1], rgb_img.shape[0]))
    result = cv2.addWeighted(rgb_img, 1, newwrap, 0.3, 0)
    cv2.putText(result, 'Vehicle is' + str(round(offset, 3)) + 'm' + side_pos + 'of center',
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
    cv2.putText(result, 'Radius of curvature:' + str(round(average_radCurv)) + 'm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), thickness=2)
    if showMe:
        plt.title('Final Result')
        plt.imshow(result)
        plt.axis('off')
        plt.show()

##########################################################################
###### https://blog.csdn.net/baidu_39415947/article/details/80367630 #####
##########################################################################