from detect_changed_segment import detect_popup_adaptive
import cv2

if __name__ == "__main__":
    image1_path = '/Users/jacobmccaughrin/github_get_changed_segement/Screenshot 2025-09-30 at 11.19.08 AM 1.png'
    image2_path = '/Users/jacobmccaughrin/github_get_changed_segement/Screenshot 2025-09-30 at 11.19.18 AM.png'

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    save_paths = ['heat_map.png', 'changed_image.png']
    print(detect_popup_adaptive(image1, image2, save_paths))