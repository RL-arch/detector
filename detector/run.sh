 #-*- coding: UTF-8 -*-
echo 'run program start'

cd bayersian
python detect_1.py
cd .. && cd yolov7
python detect_2.py

echo 'run program finish'
