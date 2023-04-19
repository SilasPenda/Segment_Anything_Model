# Segment-Anything-Model-OpenVINO

Clone this repo and cd into it

Create a Virtual environment.
```
python -m venv .venv
```

Activate Virtual environment
```
.\.venv\scripts\activate
```

Install requirements
```
pip install \
'git+https://github.com/facebookresearch/segment-anything.git'
pip install -q roboflow supervision
wget -q \
'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
pip install -q "segment_anything" "gradio>=3.25"
pip install -r openvino_requirements.txt
```

Run inference on images
```
python SAM.py --source images\truck.jpg --save result.jpg
```

Run inference on videos
```
python SAM.py --source vehicles.mp4 --save result.mp4
```

Run inference on images using OpenVINO
```
python SAM_openvino.py --source images\truck.jpg --save result.jpg
```

Run inference on videos using OpenVINO
```
python SAM_openvino.py --source vehicles.mp4 --save result.mp4
```
