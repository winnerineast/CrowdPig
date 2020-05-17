import cv2
import json


image_files_path = '/media/nvidia/red/CrowdHuman/Images/'
detect_result_file = './outputs/eval_dump/dump-13.json'

with open(detect_result_file, "r") as read_file:
    for aline in read_file:
        #aline = read_file.readline()
        print(aline)
        one_image_result = json.loads('['+aline+']')
        print('Load image '+ image_files_path + one_image_result[0]['ID'] + '.jpg')
        img = cv2.imread(image_files_path + one_image_result[0]['ID'] + '.jpg', 0)
        boxes = one_image_result[0]['dtboxes']
        if len(boxes) > 0:
            for box in boxes:
                cv2.rectangle(img, (int(box['box'][0]), int(box['box'][1])), (int(box['box'][2]), int(box['box'][3])), (0,127,128),2)

        cv2.imshow("result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()