import cv2
import time
import matplotlib.pyplot as plt
import argparse

from utils.SAM_utils import(automatic_mask_generation, show_anns)

# Create arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source', default=0, required=True, help='path to input source file') #.mp4, .m4v, .jpg, .jpeg, .png
parser.add_argument('--save', required=True, help='output path to save results to') # .mp4, .m4v, .jpg, .jpeg, .png
args = parser.parse_args()


if args.source.endswith('.jpg' or '.jpeg' or '.png'):
    image = cv2.imread(args.source)

    prediction = automatic_mask_generation(image)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(prediction)
    plt.axis('off')
    plt.savefig(args.save, bbox_inches='tight')

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
        
        prediction = automatic_mask_generation(frame)

        plt.figure(figsize=(20,20))
        plt.imshow(frame)
        show_anns(prediction)
        plt.axis('off')
        plt.savefig("results.jpg", bbox_inches='tight')
        
        image = cv2.imread("result.jpg")
        
        writer.write(image)
        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
                
    cap.release()
    cv2.destroyAllWindows()