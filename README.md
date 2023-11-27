# AI-Proctoring
The objective of this project is to develop a simplified Online Exam Proctoring System focusing on head pose angles and eye gaze estimation using webcam video. The system aims to monitor and identify behaviors such as looking away from the screen, which may indicate potential cheating during online exams.<br>
Ten CSV files, each containing data from a different person, were created, resulting in a total of more than 100,000 labelled data rows. These individual files were combined into one comprehensive CSV, which served as the input for training various machine learning models.

## Results
<ul>
<li>RandomForestClassifier: 99.49% - 4m 44s</li>
<li>SVC (Support Vector Classifier): 96.78% - 24m 34s</li>
<li>LogisticRegression: 95.05% - 2m 45s</li>
<li>MLPClassifier (Multi-layer Perceptron Classifier): 99.65% - 5m 15s</li>
<li>DecisionTreeClassifier: 98.19% - 2m 52s</li>
<li>XGBClassifier (XGBoost Classifier): 99.69% - 26s</li>
<li>GradientBoostingClassifier: 97.64% - 36m 15s</li>
</ul>

### Model Comparison:
The models were assessed based on both speed/time efficiency and accuracy. Among the models, the XGBClassifier emerged as the most efficient in terms of processing speed while maintaining a satisfactory level of accuracy for head pose angles and eye gaze estimation.
### Feasibility Considerations:
While the models exhibited promising performance, it's crucial to acknowledge the practical constraints. The current GitHub repositories for obtaining head pose angles and eye gaze estimation have limitations in terms of data acquisition speed. These repositories can capture data at a maximum rate of 3 frames per second, significantly lower than the typical 30 frames per second of webcam video. Additionally, the pipeline involves data processing and machine learning model inference, contributing to further latency.
### Active Cheating Detection:
Considering the constraints mentioned above, the system may not be suitable for real-time active cheating detection during online exams. The inherent lag in data acquisition and processing may lead to delays in identifying and responding to cheating behaviors promptly. Future improvements may involve optimizing the data acquisition process or exploring alternative models to enhance the system's real-time capabilities.


## Set up
```bash
cd AI_Proctoring
git clone https://github.com/glefundes/mobile-face-gaze.git
mv eye_angles.py mobile-face-gaze/
git clone https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino.git
mv head_angles.py HeadPoseEstimation-WHENet-yolov4-onnx-openvino/
```

Create folders for each webcam footage, with their title as "Movie.mov" or anything else (and change it in 'frame.sh'). <br>
Change folder name in 'frame.sh', 'eye_angles.py' and 'head_angles.py' accordingly.
```bash
sh frame.sh
cd HeadPoseEstimation-WHENet-yolov4-onnx-openvino
python3 head_angles.py
cd ..
cd mobile-face-gaze
python3 eye_angles.py
cd ..
```

Clean data from 'Photos/data.txt' and 'Photos/eyes.txt' for each folder and export as CSVs in 'AI-Proctoring/', in the following order and include this as first line of CSVs:<br>
H_Roll, H_Pitch, H_Yaw, E_Pitch, E_Yaw, Cheat<br>
Run the following for each CSV (Change title in code)<br>
Make sure to set "header=True" for one CSV
```python
python3 CSVfiles.csv
```

Concat all CSVs into CSV with header.<br>
Rename it as 'data-db.csv' or change name of CSV in 'code.py'
```python
python3 code.py
```

## Credits
Salil Godbole - https://github.com/Rogerbenett<br>
Dhruv Shrimali - https://github.com/DhruvShrimali<br>
Thanks to HeadPoseEstimation-WHENet-yolov4-onnx-openvino: https://github.com/PINTO0309<br>
Thanks to mobile-face-gaze: https://github.com/glefundes

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
