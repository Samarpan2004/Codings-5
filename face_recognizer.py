# face_recognizer.py
# Usage: put known face images in ./known/<name>/*.jpg
# Run: python face_recognizer.py

import os, cv2, face_recognition, numpy as np

KNOWN_DIR = "known"
TOLERANCE = 0.5
MODEL = "hog"  # or "cnn" if GPU and dlib compiled with CUDA

def load_known():
    names, encodings = [], []
    for person in os.listdir(KNOWN_DIR):
        pdir = os.path.join(KNOWN_DIR, person)
        if not os.path.isdir(pdir): continue
        for fn in os.listdir(pdir):
            path = os.path.join(pdir, fn)
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                names.append(person); encodings.append(encs[0])
    return names, encodings

def main():
    print("Loading known faces...")
    names, encs = load_known()
    if not encs:
        print("No known faces found. Place images in ./known/<name>/")
        return
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=MODEL)
        faces_enc = face_recognition.face_encodings(rgb, boxes)
        for (top,right,bottom,left), face_encoding in zip(boxes, faces_enc):
            matches = face_recognition.compare_faces(encs, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"
            if True in matches:
                name = names[matches.index(True)]
            # scale back coordinates
            top*=4; right*=4; bottom*=4; left*=4
            cv2.rectangle(frame, (left,top),(right,bottom),(0,255,0),2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1)&0xFF==27: break
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
