from detect_changed_segment import detect_popup_adaptive
import cv2

if __name__ == "__main__":
    image1_path = '/testing-images/webpage_screenshot.png'
    image2_path = '/testing-images/webpage_screenshot1.png'

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    save_paths = ['heat_map.png', 'changed_image.png']
    print(detect_popup_adaptive(image1, image2, save_paths))
