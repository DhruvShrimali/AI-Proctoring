import os
import sys
import cv2
import torch
import utils
import argparse
import traceback
import numpy as np
from PIL import Image
from models import gazenet
from mtcnn import FaceDetector

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--weights','-w', type=str, default='models/weights/gazenet.pth')
args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu")
model = gazenet.GazeNet(device)
    
state_dict = torch.load(args.weights, map_location=device)
model.load_state_dict(state_dict)

model.eval()
face_detector = FaceDetector(device=device)

folders = ["Person1", "Person2", "Person3", "Person4", "Person5", "Person6", "Person7", "Person8", "Person9", "Person10"]
for fold in folders:
    path1 = "../"+fold+"/Photos/"
    path3 = path1+"eyes.txt"
    i=1
    path2 = path1+str(i)+".jpg"
    fo=open(path3, "a+")
    while(os.path.isfile(path2)):  
        try:
            frame=cv2.imread(path2)
            frame = frame[:,:,::-1]
            frame = cv2.flip(frame, 1)
            # img_h, img_w, _ = np.shape(frame)
            # Detect Faces
            faces, landmarks = face_detector.detect(Image.fromarray(frame))
            data = fold + ": " +str(i)+"0.0000, 0.0000"+"\n"
            if len(faces) != 0:
                for f, lm in zip(faces, landmarks):
                    # Confidence check
                    if(f[-1] > 0.98):
                        # Crop and normalize face Face
                        face, gaze_origin, M  = utils.normalize_face(lm, frame)
                                            
                        # Predict gaze
                        with torch.no_grad():
                            gaze = model.get_gaze(face)
                            gaze = gaze[0].data.cpu()   
                            # print(gaze)   

                        dx = np.sin(gaze[1])
                        dy = np.sin(gaze[0])
                        format_x = "{:.4f}".format(dx)
                        format_y = "{:.4f}".format(dy)
                        data = fold + ": " +str(i)+", "+str(format_x)+", "+str(format_y)+"\n"  

        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()             
            cv2.destroyAllWindows()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                limit=2, file=sys.stdout)
            data = fold + ": " +str(i)+", 0.0000, 0.0000"+"\n"
            fo.write(data)
            if(i%100==0):
                print(i)
            i=i+1
            path2 = path1+str(i)+".jpg"
            continue

        # print(data)
        fo.write(data) 
        if(i%100==0):
                print(i)
        i=i+1
        path2 = path1+str(i)+".jpg"    
    fo.close()
    cv2.destroyAllWindows()
cv2.destroyAllWindows()