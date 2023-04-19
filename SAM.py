import cv2
import torch
import supervision as sv
import time
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import argparse

# Create arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='path to input source file') #.mp4, .m4v, .jpg, .jpeg, .png
parser.add_argument('--save', required=True, help='output path to save results to') # .mp4, .m4v, .jpg, .jpeg, .png
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
MODEL_PATH = "sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)




if args.source.endswith('.jpg' or '.jpeg' or '.png'):
    image_bgr = cv2.imread("horse.jpg")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = mask_generator.generate(image_rgb)

    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(image_bgr, detections)

    cv2.imwrite("output.jpg", annotated_image)
    
else:
    # Run this for inference on videos or webcam
    cap = cv2.VideoCapture(args.source)

    # Get the Default resolutions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc  = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Define the codec and filename.
    writer = cv2.VideoWriter(args.save, fourcc, cap_fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print('No frame read!')
            break
        
        image_bgr = cv2.imread("horse.jpg")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = mask_generator.generate(image_rgb)

        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(result)
        annotated_image = mask_annotator.annotate(image_bgr, detections)
        
        writer.write(annotated_image)
       
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
                
    cap.release()
    cv2.destroyAllWindows()