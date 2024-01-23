import warnings
warnings.simplefilter('ignore')
from mobile_sam import sam_model_registry, SamPredictor
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
matplotlib.use('tkagg')


def convert_bbox2xyxy(results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
    input_box = np.array(results[0].boxes.xyxy)
    return input_box


def remove_small_cnt(masks_final):
    contours, hierarchy = cv2.findContours(masks_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour
    bigger = max(contours, key=lambda item: cv2.contourArea(item))

    # Filter small contours
    contours_final = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > cv2.contourArea(bigger) / 5:
            contours_final.append(contours[i])
    return contours_final


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# configure device and input image
device = "cpu"
path = 'data/leaf_test/leaf_test2.jpg'
image = cv2.imread(path).astype(np.uint8)
image = cv2.resize(image, (640, 480))
# load model
model = YOLO('weights/best.pt')
# set model parameters
model.overrides['conf'] = 0.4  # NMS confidence threshold
model.overrides['iou'] = 0.2  # NMS IoU threshold
model.overrides['agnostic_nms'] = True  # NMS class-agnostic
model.overrides['max_det'] = 5  # maximum number of detections per image

model_type = "vit_t"
sam_checkpoint = "weights/mobile_sam1.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# mobile_sam.eval()
predictor = SamPredictor(mobile_sam)
predictor.set_image(image)
# set image

# perform inference
results = model.predict(image)
input_box = convert_bbox2xyxy(results)

centers = np.zeros((input_box.shape[0], 2))
for i, box in enumerate(input_box):
    center_x, center_y = box[0] / 2 + box[2] / 2, box[1] / 2 + box[3] / 2
    centers[i, :] = np.array([center_x, center_y])


masks_final = []
for i, center in enumerate(centers):
    masks, scores, logits = predictor.predict(
        point_coords=center.reshape(1, 2),
        box=input_box[i],
        point_labels=[1],
        multimask_output=True)
    masks = (np.moveaxis(masks, 0, -1)).astype(np.uint8)
    best_mask = masks[:, :, np.argmax(scores)]
    if i == 0:
        masks_final = best_mask
    else:
        masks_final += best_mask



cv2.imshow("binary2", masks_final)

contours_final = remove_small_cnt(masks_final)
# Create a new binary mask with only the biggest contour
new_mask = np.zeros_like(masks_final)
masks = cv2.drawContours(new_mask, contours_final, -1, (255, 255, 255), thickness=cv2.FILLED).astype(np.uint8)

cv2.drawContours(masks, contours_final, -1, (255, 255, 255), cv2.FILLED)
cv2.drawContours(image=image, contours=contours_final, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

# see the results
cv2.imshow("binary2", masks)
# print(masks.shape, image.shape)
img = cv2.bitwise_and(image, image, mask=masks)
cv2.imshow('frame', img)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the image to Torch tensor
img_tensor = torch.tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
img_tensor = F.convert_image_dtype(img_tensor, dtype=torch.uint8).permute(2, 0, 1)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

drawn_boxes = draw_bounding_boxes(img_tensor, results[0].boxes.xyxy, colors="red")
show(drawn_boxes)
plt.show()
plt.savefig("test1.jpg")
# # # observe results
# # print(results[0].boxes)
# # render = render_result(model=model, image=image, result=results[0])
# # render.show()
