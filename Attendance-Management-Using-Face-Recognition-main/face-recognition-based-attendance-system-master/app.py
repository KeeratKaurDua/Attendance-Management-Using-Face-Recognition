import cv2
import os
import json
from flask import Flask, request, render_template, render_template_string
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)

nimgs = 10
MODEL_PATH = 'static/face_recognition_model.pkl'
METRICS_PATH = 'static/model_metrics.json'

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load(MODEL_PATH)
    return model.predict(facearray)


# ----------------------------------------------------------------------
# MODEL EVALUATION  (the 4 metrics)
#   1. Accuracy   - share of test faces identified correctly
#   2. Precision  - when the model says "this is X", how often it is right
#   3. Recall     - of all of X's faces, how many the model finds
#   4. F1-score   - harmonic mean of precision and recall
# Precision, recall and F1 use weighted averaging across users.
# ----------------------------------------------------------------------
def evaluate_model(X, y):
    """Hold out 20% of the images, fit a temporary KNN on the rest and
    score it on the held-out images. Returns a dict, or None if there is
    not enough data (needs 2+ users and 2+ images per user)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        return None

    knn = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        'train_images': int(len(X_train)),
        'test_images': int(len(X_test)),
        'users': int(len(classes)),
        'evaluated_at': datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
    }


def save_metrics(metrics):
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def metrics_message(metrics):
    if metrics is None:
        return 'Model trained. Not enough data to compute metrics (need 2+ users).'
    return (f"Model trained. Accuracy: {metrics['accuracy']:.2%} | "
            f"Precision: {metrics['precision']:.2%} | "
            f"Recall: {metrics['recall']:.2%} | "
            f"F1: {metrics['f1_score']:.2%}")


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            if img is None:
                continue  # skip unreadable files
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    if len(faces) == 0:
        raise ValueError('No face images available to train on.')

    faces = np.array(faces)
    labels = np.array(labels)

    # 1) Evaluate on a held-out split to get honest metrics
    metrics = evaluate_model(faces, labels)
    if metrics is not None:
        save_metrics(metrics)
    elif os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)  # stale metrics no longer apply

    # 2) Train the final model on ALL images and save it
    knn = KNeighborsClassifier(n_neighbors=min(5, len(faces)))
    knn.fit(faces, labels)
    joblib.dump(knn, MODEL_PATH)

    return metrics


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser + '/' + i)
    os.rmdir(duser)


################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)

    ## if all the faces are deleted, delete the trained model and metrics
    if os.listdir('static/faces/') == []:
        for path in (MODEL_PATH, METRICS_PATH):
            if os.path.exists(path):
                os.remove(path)
    else:
        try:
            train_model()  # retrains and refreshes the metrics
        except Exception:
            pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Metrics page: shows the latest evaluation of the face-recognition model
METRICS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Model Metrics</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f4f6fb; margin:0; padding:40px; }
    .card { max-width:620px; margin:auto; background:#fff; padding:30px; border-radius:12px;
            box-shadow:0 2px 10px rgba(0,0,0,.1); }
    h2 { margin-top:0; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:20px 0; }
    .metric { background:#f0f3ff; border-radius:10px; padding:18px; text-align:center; }
    .metric .value { font-size:28px; font-weight:bold; color:#5620fb; }
    .metric .label { color:#555; margin-top:6px; }
    .meta { color:#666; font-size:14px; line-height:1.7; }
    a { color:#5620fb; text-decoration:none; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Face Recognition Model Metrics</h2>
    {% if m %}
      <div class="grid">
        <div class="metric"><div class="value">{{ '%.2f'|format(m.accuracy * 100) }}%</div><div class="label">Accuracy</div></div>
        <div class="metric"><div class="value">{{ '%.2f'|format(m.precision * 100) }}%</div><div class="label">Precision</div></div>
        <div class="metric"><div class="value">{{ '%.2f'|format(m.recall * 100) }}%</div><div class="label">Recall</div></div>
        <div class="metric"><div class="value">{{ '%.2f'|format(m.f1_score * 100) }}%</div><div class="label">F1-score</div></div>
      </div>
      <div class="meta">
        Registered users: {{ m.users }}<br>
        Training images: {{ m.train_images }} &nbsp;|&nbsp; Test images: {{ m.test_images }}<br>
        Last evaluated: {{ m.evaluated_at }}<br>
        Precision, recall and F1 are weighted averages across users.
      </div>
    {% else %}
      <p>No metrics available yet. Register at least 2 users so the model can be evaluated.</p>
    {% endif %}
    <p><a href="/">&larr; Back to home</a></p>
  </div>
</body>
</html>
"""


@app.route('/metrics')
def metrics():
    return render_template_string(METRICS_TEMPLATE, m=load_metrics())


# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No trained model found. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if frame is not captured properly

        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)

            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)

            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Attendance', frame)

        # Close window when pressing 'Esc' (27) or clicking the red cross
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Attendance', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()  # Close the webcam window properly

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if frame is not captured properly

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1

        if j == nimgs * 5:
            break

        cv2.imshow('Adding new User', frame)

        if cv2.waitKey(1) == 27:  # Press 'Esc' to close
            break

    cap.release()
    cv2.destroyAllWindows()  # Close the webcam window

    print('Training Model')
    metrics = train_model()
    print(metrics_message(metrics))

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=metrics_message(metrics))


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
