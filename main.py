from PyQt5.QtWidgets import QGraphicsOpacityEffect
import cv2.data
import matplotlib.pyplot as plt
import cv2.data
from PyQt5.QtWidgets import QGroupBox
import numpy as np
from PIL import ImageGrab
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QProgressBar
from torch import nn
from PyQt5.QtWidgets import QTextEdit, QFrame
import fitz
from PyQt5.QtGui import QPainter,  QBrush
from PyQt5.QtWidgets import QLineEdit,  QCheckBox
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QScrollBar
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QMovie
import cv2
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QMainWindow, QPushButton, QSlider
from PyQt5.QtCore import QUrl, QDir, QTimer
from PyQt5.QtGui import QFont, QColor
import torch
from albumentations import Compose, CenterCrop
from albumentations.pytorch import ToTensorV2
from models.MAT import MAT
from face_detection.Face_dect import main
import time
from config import train_config
import threading
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QScrollArea, QLabel, QVBoxLayout, QDialog, QFileDialog, \
    QMenu
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
import pygame
from dashscope import Generation

dashscope.api_key = 'sk-079a674030fb4bfd9fd03ea551220ac8'

'''

加载模型界面
1.初始化模型
2.加载成功后自动进入登陆界面

'''

class ModelLoader(QMainWindow):
    loadingCompleted = pyqtSignal()  # 定义加载完成的信号

    def __init__(self):
        super().__init__()
        self.setFixedSize(600, 400)  # 增加高度以容纳图片
        self.setWindowIcon(QIcon('favicon.ico'))

        layout = QVBoxLayout()
        # 添加图片
        image_label = QLabel()
        pixmap = QPixmap('./images/图片1.png')
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)  # 缩放图片以适应标签大小
        image_label.setStyleSheet("border-radius: 10px; border: 4px solid #000033;")  # 设置圆角和边框
        layout.addWidget(image_label)

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 设置范围为不确定，即流动进度条
        layout.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 设置整体样式表
        self.setStyleSheet("""
            QWidget {
                background-color: #000000; /* 设置深色背景色 */
                color: white; /* 设置文本颜色为白色 */
            }
            QProgressBar {
                border: 2px solid #CCFFFF;
                border-radius: 10px;
                background-color: #000033; /* 设置进度条的背景色 */
            }
            QProgressBar::chunk {
                background-color: #00FFFF; /* 设置进度条的颜色 */
                width: 6px; /* 设置进度条块的宽度 */
            }
            QMenuBar {
                background-color: transparent; /* 设置菜单栏透明 */
            }
            QMenuBar::item {
                background-color: transparent; /* 设置菜单项透明 */
            }
            QMainWindow::title {
                background-color: transparent; /* 设置标题栏透明 */
            }
        """)

        # 使窗口标题栏透明
        self.setWindowFlags(Qt.FramelessWindowHint)

        QTimer.singleShot(100, self.load_model_thread)

        # 初始化模型和人脸融合模型
        self.model = None
        self.image_face_fusion = None


    def load_model_thread(self):
        # 创建新的线程来加载模型
        model_thread = threading.Thread(target=self.load_model)
        model_thread.start()

    def load_model(self):
        if (panduan == False):
            model_dict_c23 = torch.load(r"model_dict/model_best_c23", map_location=torch.device('cpu'))
            model_dict_c40 = torch.load(r"model_dict/model_best_c40", map_location=torch.device('cpu'))
        elif (panduan == True):
            model_dict_c23 = torch.load(r"model_dict/model_best_c23")
            model_dict_c40 = torch.load(r"model_dict/model_best_c40")
        feature_layer = 'b2'
        name = 'EFB4_ALL_c23_trunc_b2'
        Config = train_config(name, ['ff-all-c23', 'efficientnet-b4'], attention_layer='b5',
                              feature_layer=feature_layer, epochs=20, batch_size=4, augment='augment1')
        self.model_23 = MAT(**Config.net_config)
        self.model_23.load_state_dict(model_dict_c23, strict=False)
        self.model_23 = self.model_23.to('cuda').eval()
        self.model_40 = MAT(**Config.net_config)
        self.model_40.load_state_dict(model_dict_c40, strict=False)
        self.model_40 = self.model_40.to('cuda').eval()


        # # 初始化人脸融合模型
        # self.image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')

        # 发送加载完成的信号
        self.loadingCompleted.emit()

    def show_login_window(self):
        # 创建登录窗口并显示
        self.close()
        self.yuanwuyuan = LoginWindow(loader)
        self.yuanwuyuan.show()

    def closeEvent(self, event):
        event.accept()

    def predict_main_videos_23(self, image):
        img = np.array(image)
        resize = (380, 380)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Compose([CenterCrop(*resize), ToTensorV2()])(image=img)['image']
        img = img.to(torch.float32).to('cuda', non_blocking=True)
        img = torch.reshape(img, [1, 3, 380, 380])
        forecast = self.model_23(img, train_batch=False)
        probabilities= nn.Softmax(dim=1)(forecast)
        label = torch.max(forecast, dim=1)[1]
        if label == 0:
            return "Fake", round(probabilities[0][0].item(), 3)  # 返回 "Fake" 类别的概率，并保留两位小数
        else:
            return "Real", round(probabilities[0][1].item(), 3)  # 返回 "Real" 类别的概率，并保留两位小数

    def predict_main_videos_40(self, image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize = (380, 380)
        img = Compose([CenterCrop(*resize), ToTensorV2()])(image=img)['image']
        img = img.to(torch.float32).to('cuda', non_blocking=True)
        img = torch.reshape(img, [1, 3, 380, 380])
        forecast = self.model_40(img, train_batch=False)
        probabilities= nn.Softmax(dim=1)(forecast)
        label = torch.max(forecast, dim=1)[1]
        if label == 0:
            return "Fake", round(probabilities[0][0].item(), 3)  # 返回 "Fake" 类别的概率，并保留两位小数
        else:
            return "Real", round(probabilities[0][1].item(), 3)  # 返回 "Real" 类别的概率，并保留两位小数


def generate_explanation(context, question="这个视频有伪造吗？"):
    prompt = f'''请基于```内的内容回答问题。"
    ```
    {context}
    ```
    我的问题是：{question}。
    '''
    rsp = Generation.call(model='qwen-turbo', prompt=prompt)
    return rsp.output.text


###########################################################################################################################
##########################################################################################################################
#######################################################第二页################################################################

class SecondWindow(QWidget):
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.predict_23 = self.model_loader.predict_main_videos_23
        self.predict_40 = self.model_loader.predict_main_videos_40

        self.ModelFlag = 0
        super().__init__()

        self.flag = 0

        self.setWindowTitle('视频 测试')

        self.real_count_1 = 0
        self.total_count_1 = 0
        self.frame_counter = 0
        self.frame_number = 0
        self.frame_fake = 0
        self.frame_ture = 0

        self.setFixedSize(int(2560*0.9), int(0.9*1440))  # 设置固定的窗口大小
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.move(140, 100)
        self.setWindowIcon(QIcon('favicon.ico'))
        self.frame_count = 0
        self.video_path = ""
        self.fps = 0

        self.btn_minimize = QPushButton("-", self)
        self.btn_minimize.setGeometry(2200, 0, 50, 50)
        self.btn_minimize.setStyleSheet("background-color: transparent;font-size: 12pt; border: none; color: white;font-weight: bold;")
        self.btn_minimize.clicked.connect(self.minimize)

        self.btn_close = QPushButton("×", self)
        self.btn_close.setGeometry(2250, 0, 50, 50)
        self.btn_close.setStyleSheet("background-color: transparent;font-size: 12pt; border: none; color: white;font-weight: bold;")
        self.btn_close.clicked.connect(self.close)

        # 设置表格的样式
        self.table_video = QTableWidget(self)
        self.table_video.setGeometry(1400, 950, 440, 240)
        self.table_video.setColumnCount(2)
        self.table_video.setHorizontalHeaderLabels(["真伪", "置信度"])

        self.table_video.setStyleSheet("""
            QTableWidget {
                background-color: transparent;  /* 背景色设置为透明 */
                alternate-background-color: transparent;  /* 交替背景色设置为透明 */
                border: 4px solid #003472;
                color: white;  /* 修改字体颜色为白色 */
                border-radius: 6px;
                font-weight: bold;
            }
            QHeaderView::section {
                background-color: transparent;  /* 设置标题栏背景为透明 */
                color: black;  /* 修改标题栏字体颜色为黑色 */
                border: 2px solid #003472;
                font-weight: bold;
                padding: 4px;  /* 添加内边距使标题栏看起来更大气 */
            }
            QHeaderView::item {
                background-color: transparent;  /* 设置标题栏项目背景为透明 */
                color: black;  /* 修改标题栏项目字体颜色为黑色 */
                border: 2px solid #003472;
                font-weight: bold;
            }
            QTableWidget::item {
                background-color: transparent;  /* 设置单元格背景为透明 */
                color: white;  /* 修改单元格字体颜色为白色 */
                border: 2px solid #003472;
            }
            QTableWidget::item:selected {
                background-color: #005bb5;  /* 设置选中单元格的背景色 */
                color: white;  /* 设置选中单元格的字体颜色 */
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 12px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)

        # 创建一个标签用于显示图片
        self.table_back_label = QLabel(self)
        self.table_back_label.setGeometry(1880, 970, int(196*1.2), int(165*1.2))

        # 加载图片
        pixmap_table = QPixmap('./images/biaoge.png')
        self.table_back_label.setPixmap(pixmap_table)
        self.table_back_label.setScaledContents(True)

        # 创建一个标签用于显示视频帧
        self.video_label = QLabel(self)
        self.video_label.setGeometry(585, 125, int(0.9*640)+60, int(0.9*480)-25)

        # 创建 QLabel 作为边框,视频播放显示区域
        self.border_label = QLabel(self)
        self.border_label.setGeometry(610, 100, int(0.9*644), int(0.9*484))
        self.border_label.setStyleSheet("background-color: transparent;")

        # 创建 QLabel 作为边框,视频播放显示区域
        self.result_label = QLabel(self)
        self.result_label.setGeometry(783, 640, 400, 120)
        self.result_label.setStyleSheet("background-color: transparent;")

        # 开启拖拽功能
        self.setAcceptDrops(True)

        self.tuozhuai_label = QLabel("将文件拖拽到此处", self)
        self.tuozhuai_label.setGeometry(790, 450, 200, 30)
        self.tuozhuai_label.setStyleSheet("background-color: transparent ; font-size: 12pt; font-weight: bold; color: white ; font-family: 方正姚体;")
        self.tuozhuai_label.lower()

        #人脸捕获区域
        self.new_label = QLabel(self)
        self.new_label.setGeometry(1570, 185, 420, 460)
        self.new_label.setStyleSheet("background-color: transparent; border: 4px solid #2e4e7e; border-radius: 6px;")

        # 创建水平布局用于容纳帧
        self.frame_layout = QHBoxLayout()

        # 创建滚动区域来容纳水平布局
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setGeometry(120, 880, 940, 300)

        # 设置滚动区域的样式表
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }"
                                       "QScrollBar:vertical { width: 8px; background: transparent; border-radius: 4px; }"
                                       "QScrollBar::handle:vertical { background: #808080; border-radius: 4px; }"
                                       "QScrollBar::add-line:vertical { background: transparent; }"
                                       "QScrollBar::sub-line:vertical { background: transparent; }"
                                       "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }")

        # 在滚动区域内设置水平布局
        content_widget = QWidget()
        content_widget.setLayout(self.frame_layout)

        # 设置内容小部件的样式表，使其背景透明
        content_widget.setStyleSheet("background-color: transparent;")

        self.scroll_area.setWidget(content_widget)

        # 创建一个标签用于显示图片
        self.content_widget_label = QLabel(self)
        self.content_widget_label.setGeometry(160, 880, 780, 286)

        # 加载图片
        pixmap_content_widget = QPixmap('./images/content_widget.png')
        self.content_widget_label.setPixmap(pixmap_content_widget)
        self.content_widget_label.setScaledContents(True)



        # 添加四个区域
        self.region1 = QLabel(self)
        self.region1.setGeometry(200+300-150, 15, 180, 40)
        self.region1.setText("实时 平台")
        self.region1.setStyleSheet("background-color: transparent;color:white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region1.mousePressEvent = lambda event: self.on_region_clicked(1)

        self.region2 = QLabel(self)
        self.region2.setGeometry(300+600-150, 15, 180, 40)
        self.region2.setStyleSheet("background-color: transparent;color: white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region2.mousePressEvent = lambda event: self.on_region_clicked(2)
        self.region2.setText("视频 检测")

        self.region3 = QLabel(self)
        self.region3.setGeometry(400+900-150, 15, 180, 40)
        self.region3.setStyleSheet("background-color: transparent;color: white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region3.mousePressEvent = lambda event: self.on_region_clicked(3)
        self.region3.setText("使用 说明")

        self.region4 = QLabel(self)
        self.region4.setGeometry(500+1200-150, 15, 180, 40)
        self.region4.setStyleSheet("background-color: transparent;color: white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region4.mousePressEvent = lambda event: self.on_region_clicked(4)
        self.region4.setText("团队 介绍")

        self.region5 = QLabel(self)
        self.region5.setGeometry(600+1500-150, 15, 180, 40)
        self.region5.setStyleSheet("background-color: transparent;color: white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region5.mousePressEvent = lambda event: self.on_region_clicked(5)
        self.region5.setText("工具")

        #给每个区域设置蒙版
        opacity_effect1 = QGraphicsOpacityEffect()
        opacity_effect1.setOpacity(0.33)

        opacity_effect2 = QGraphicsOpacityEffect()
        opacity_effect2.setOpacity(1.00)

        opacity_effect3 = QGraphicsOpacityEffect()
        opacity_effect3.setOpacity(0.33)

        opacity_effect4 = QGraphicsOpacityEffect()
        opacity_effect4.setOpacity(0.33)

        opacity_effect5 = QGraphicsOpacityEffect()
        opacity_effect5.setOpacity(0.33)

        font_pro_region = QFont("黑体", 18, QFont.Bold)
        self.region1.setFont(font_pro_region)
        self.region2.setFont(font_pro_region)
        self.region3.setFont(font_pro_region)
        self.region4.setFont(font_pro_region)
        self.region5.setFont(font_pro_region)

        self.region1.setGraphicsEffect(opacity_effect1)
        self.region2.setGraphicsEffect(opacity_effect2)
        self.region3.setGraphicsEffect(opacity_effect3)
        self.region4.setGraphicsEffect(opacity_effect4)
        self.region5.setGraphicsEffect(opacity_effect5)


        self.yuanwuyuan = 0
        self.initUII()

    def initUII(self):
        self.pushButton_Add = QPushButton('添加视频', self)
        self.pushButton_Add.setGeometry(20+250, 220, 140, 50)
        self.pushButton_Add.setStyleSheet("background-color: transparent;color:white;font-weight: bold; font-size: 16pt;font-family: 优设标题黑;")
        self.pushButton_Add.clicked.connect(self.on_pushButton_Add_clicked)

        self.pushButton_model1 = QPushButton('模型 一', self)
        self.label_model1 = QLabel('检测 高清视频', self)
        self.pushButton_model1.setGeometry(20+250, 395, 140, 50)
        self.label_model1.setGeometry(23+250, 432, 140, 50)
        self.pushButton_model1.setStyleSheet("background-color: transparent;color:white;font-weight: bold; font-size: 16pt;font-family: 优设标题黑;")
        self.label_model1.setStyleSheet("background-color: transparent;color:white;font-weight: bold; font-size: 10pt;font-family: 方正姚体;")
        self.pushButton_model1.clicked.connect(self.on_pushButton_Add_model_23)

        self.pushButton_model2 = QPushButton('模型 二', self)
        self.label_model2 = QLabel('检测 低清视频', self)
        self.pushButton_model2.setGeometry(20+250, 580, 140, 50)
        self.label_model2.setGeometry(23+250, 617, 140, 50)
        self.pushButton_model2.setStyleSheet("background-color: transparent;color:white;font-weight: bold; font-size: 16pt;font-family: 优设标题黑;")
        self.label_model2.setStyleSheet("background-color: transparent;color:white;font-weight: bold; font-size: 10pt;font-family: 方正姚体;")
        self.pushButton_model2.clicked.connect(self.on_pushButton_Add_model_40)

        self.pushButton_player = QPushButton(QIcon("./images/620.bmp"), '播放', self)
        self.pushButton_player.setGeometry(180+570, 560, 80, 50)
        self.pushButton_player.setStyleSheet("background-color: transparent;color:white;font-weight: bold; font-size: 12pt;font-family: 方正姚体;")
        self.pushButton_player.clicked.connect(self.on_pushButton_Player_clicked)

        self.pushButton_pause = QPushButton(QIcon("./images/622.bmp"), '暂停', self)
        self.pushButton_pause.setGeometry(280+570, 560, 80, 50)
        self.pushButton_pause.setStyleSheet("background-color: transparent; color:white;font-weight: bold; font-size: 12pt;font-family: 方正姚体;")
        self.pushButton_pause.clicked.connect(self.on_pushButton_Pause_clicked)

        self.pushButton_stop = QPushButton(QIcon("./images/624.bmp"), '重置', self)
        self.pushButton_stop.setGeometry(380+570, 560, 80, 50)
        self.pushButton_stop.setStyleSheet("background-color: transparent; color:white;font-weight: bold; font-size: 12pt;font-family: 方正姚体;")
        self.pushButton_stop.clicked.connect(self.on_pushButton_Stop_clicked)

        # 进度条
        self.label_2 = QLabel('0:0/0:0', self)
        font = QFont("Arial", 12)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: white")
        self.label_2.setGeometry(1135, 608, 250, 30)

        self.horizontalSlider_2 = QSlider(Qt.Horizontal, self)
        self.horizontalSlider_2.setGeometry(670, 608, 450, 30)
        self.horizontalSlider_2.setRange(0, 100)
        self.horizontalSlider_2.sliderMoved.connect(self.on_slider_moved)
        # 设置滑块的样式
        self.horizontalSlider_2.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #999999;
                        height: 10px;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                        margin: 2px 0;
                    }

                    QSlider::handle:horizontal {
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0077CC, stop:1 #0055CC);
                        border: 1px solid #5c5c5c;
                        width: 18px;
                        margin: -2px 0; 
                        border-radius: 3px;
                    }
                """)

        # 创建一个标签用于显示 GIF 图像
        self.gif_label = QLabel(self)
        self.gif_label.setGeometry(1350-580, 650, 120, 120)

        self.giftext_label = QLabel(self)
        self.giftext_label.setGeometry(1520-600, 650, 120, 120)
        self.giftext_label.setText("检测中")
        self.giftext_label.setStyleSheet(
            "background-color: transparent ; font-size: 16pt; font-weight: bold; color: white ; font-family: 优设标题黑;")

        # 加载 GIF 图像
        self.movie = QMovie('./images/6.gif')
        self.gif_label.setMovie(self.movie)
        self.gif_label.lower()

        self.gif_label.hide()
        self.giftext_label.hide()

        # 开始播放 GIF 图像
        self.movie.start()
        self.gif_label.setStyleSheet("background-color: transparent;")

        # 创建标签并插入图片
        image_label = QLabel(self)
        pixmap = QPixmap("./images/Final_1.png")  # 替换为您的图片路径
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)  # 缩放图片以适应标签大小
        image_label.setGeometry(0, 0, int(2560*0.9), int(0.9*1440))
        image_label.lower()

        self.timer_video = QTimer(self)
        self.timer_video.timeout.connect(self.update_slider_position)

        # 创建一个定时器用于更新视频帧
        self.timer_else = QTimer(self)
        self.timer_else.timeout.connect(self.update_frame_video_else)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame_video)

    def on_region_clicked(self, region_number):

        if region_number == 1:
            self.close()
            self.First = HomeWindow(self.model_loader)
            self.First.show()
            pass

        elif region_number == 2:
            pass

        elif region_number == 3:
            opacity_effect3 = QGraphicsOpacityEffect()
            opacity_effect3.setOpacity(1.00)
            self.region3.setGraphicsEffect(opacity_effect3)
            self.thirdwindow = PDFViewer()
            self.thirdwindow.show()

            pass

        elif region_number == 4:
            opacity_effect4 = QGraphicsOpacityEffect()
            opacity_effect4.setOpacity(1.00)
            self.region4.setGraphicsEffect(opacity_effect4)
            self.fourthwindow = ImageDisplay()
            self.fourthwindow.show()
            pass

        elif region_number == 5:
            opacity_effect5 = QGraphicsOpacityEffect()
            opacity_effect5.setOpacity(1.00)
            self.region5.setGraphicsEffect(opacity_effect5)
            self.fivethwindow = ToolOptionsWindow()
            self.fivethwindow.show()
            pass

    def update_frame_video(self):
        if self.yuanwuyuan == 1:
            video_cap = self.cap
            self.gif_label.show()
            self.giftext_label.show()

            while True:
                ret, frame = video_cap.read()
                self.frame_counter += 1
                if not ret:
                    video_processed = True
                    break
                faces = main(frame)
                if self.yuanwuyuan == 0:
                    return
                if faces is not None:
                    x1 = int(0.95 * faces[0])
                    y1 = int(0.9 * faces[1])
                    h = int(1.2 * faces[3])
                    w = int(1.2 * faces[2])
                    x2 = w + x1
                    y2 = h + y1
                    if x1 < 0: x1 = 0
                    if x2 < 0: x2 = 0
                    if y1 < 0: y1 = 0
                    if y2 < 0: y2 = 0

                    face_img = frame[y1:y2, x1:x2]

                    face_img = cv2.resize(face_img, (380, 380))
                    self.frame_number += 1
                    frame_label = QLabel()
                    frame_label.setFixedSize(210, 250)

                    # 预测结果
                    if self.ModelFlag == 0:
                        label, probability = self.predict_23(face_img)
                    elif self.ModelFlag == 1:
                        label, probability = self.predict_40(face_img)

                    self.table_video.insertRow(self.table_video.rowCount())
                    self.table_video.setItem(self.table_video.rowCount() - 1, 0,
                                             QTableWidgetItem("{}".format(label)))
                    self.table_video.setItem(self.table_video.rowCount() - 1, 1,
                                             QTableWidgetItem("{:.2%}".format(probability)))

                    color = (0, 255, 102) if label == "Real" else (0, 0, 204)
                    cv2.putText(face_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    h, w, c = face_img_rgb.shape
                    qimg = QImage(face_img_rgb.data, w, h, w * c, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg).scaled(420, 460, Qt.IgnoreAspectRatio)
                    self.new_label.setPixmap(pixmap)

                    if label == "Real":
                        border_color = "rgb(0, 255, 102)"
                        self.frame_ture += 1
                    else:
                        border_color = "rgb(204, 0, 0)"
                        self.frame_fake += 1
                    self.new_label.setStyleSheet(f"border: 4px solid {border_color};")

                    pixmap_else = QPixmap.fromImage(qimg).scaled(210, 250, Qt.KeepAspectRatio)
                    self.content_widget_label.hide()

                    if self.frame_number <= 20:
                        clickable_label = ClickableLabel(pixmap_else)
                        self.frame_layout.addWidget(clickable_label)

                    QApplication.processEvents()

            # 在视频处理完毕后弹出信息框
            if video_processed:
                QMessageBox.information(self, "提示", "视频检测完毕！")
                self.gif_label.hide()
                self.giftext_label.hide()
            else:
                QMessageBox.critical(self, "错误", "视频处理中断！")
            self.yuanwuyuan = 0
            video_cap.release()
            if self.frame_number == 0:
                QMessageBox.information(self, "提示", "请重新检测！")
                return
            fake_percentage = (self.frame_fake / self.frame_number) * 100
            formatted_percentage = "{:.2f}%".format(fake_percentage)
            result_text = f"""
                <html>
                <head>
                <style>
                .result-text {{
                    font-family: Arial, sans-serif;
                    font-size: 20px;
                    line-height: 1.5;
                    color: white;  /* 设置文字颜色为白色 */
                    font-weight: bold;  /* 将所有文字加粗 */
                }}
                .result-heading {{
                    font-size: 24px;  /* 设置标题字体大小为 24px */
                }}
                .highlight-fake {{
                    font-weight: bold;
                    color: #ff0000;  /* 设置高亮文字颜色为红色 */
                }}
                .highlight-true {{
                    font-weight: bold;
                    color: #00ff00;  /* 设置高亮文字颜色为绿色 */
                }}
                </style>
                </head>
                <body>
                <div class="result-text">
                    <span class="result-heading">结果：总共检测 {self.frame_number} 帧</span><br>
                    <span class="highlight-fake">Fake：</span><span style="color: white;">{self.frame_fake}/{self.frame_number}&nbsp;&nbsp;&nbsp;&nbsp;{formatted_percentage}</span><br>
                    <span class="highlight-true">True：</span><span style="color: white;">{self.frame_ture}/{self.frame_number}&nbsp;&nbsp;&nbsp;{100 - fake_percentage:.2f}%</span>
                </div>
                </body>
                </html>
            """
            # 显示结果文本
            self.result_label.setText(result_text)
            QApplication.processEvents()

            # 生成解释文本
            explanation = generate_explanation(f'某视频经过DeepFake检测，检测如下：{result_text}')

            # 在显示文字后生成并播放语音
            tts_text = explanation
            result = SpeechSynthesizer.call(model='sambert-zhiqian-v1', text=tts_text, sample_rate=48000)

            # 设置音频文件路径，使用时间戳确保文件名唯一
            timestamp = int(time.time())
            audio_file = f'output_{timestamp}.wav'
            try:
                if result.get_audio_data() is not None:
                    # 保存新的音频文件
                    with open(audio_file, 'wb') as f:
                        f.write(result.get_audio_data())

                    # 初始化混音器并播放音频
                    pygame.mixer.init()
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()

                    # 保持事件循环直到音频播放结束
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
            except PermissionError as e:
                print(f"无法写入文件 {audio_file}，权限错误：{e}")
            except Exception as e:
                print(f"保存或播放音频文件时出错：{e}")

            # 重置帧计数器
            self.frame_counter = 0
            self.frame_number = 0
            self.frame_fake = 0
            self.frame_ture = 0
            return

    def clearImages(self):
        self.clear_scroll_area()
        self.result_label.clear()
        self.new_label.clear()
        self.new_label.setStyleSheet("background-color: transparent; border: 4px solid #2e4e7e; border-radius: 6px;")
        self.reset_scroll_area()
        # 清空表格内容
        self.table_video.clearContents()
        self.table_video.setRowCount(0)

    def reset_scroll_area(self):
        # 创建水平布局用于容纳帧
        self.frame_layout = QHBoxLayout()
        # 设置滚动区域的样式表
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }"
                                       "QScrollBar:vertical { width: 8px; background: transparent; border-radius: 4px; }"
                                       "QScrollBar::handle:vertical { background: #808080; border-radius: 10px; width: 4px; }"
                                       "QScrollBar::add-line:vertical { background: transparent; }"
                                       "QScrollBar::sub-line:vertical { background: transparent; }"
                                       "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
                                       "QScrollBar:horizontal { height: 8px; background: transparent; border-radius: 4px; }"
                                       "QScrollBar::handle:horizontal { background: #808080; border-radius: 10px; height: 4px; }"
                                       "QScrollBar::add-line:horizontal { background: transparent; }"
                                       "QScrollBar::sub-line:horizontal { background: transparent; }"
                                       "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }")
        # 在滚动区域内设置水平布局
        content_widget = QWidget()
        content_widget.setLayout(self.frame_layout)
        # 设置内容小部件的样式表，使其背景透明
        content_widget.setStyleSheet("background-color: transparent;")
        self.scroll_area.setWidget(content_widget)

    def clear_scroll_area(self):
        content_widget = self.scroll_area.takeWidget()
        if content_widget:
            content_widget.deleteLater()

    def on_pushButton_Add_clicked(self):
        currentpath = QDir.homePath()
        dlgTitle = "选择 视频路径"
        strfilter = "Mp4 Files(*.mp4);;All Files(*.*)"

        # 获取用户选择的文件路径
        self.allfiles, _ = QFileDialog.getOpenFileName(self, dlgTitle, currentpath, strfilter)

        # 检查是否选择了文件
        if not self.allfiles:
            QMessageBox.warning(self, "警告", "请选择一个视频文件！")
            return

        # 设置视频路径并打开视频文件
        self.video_path = self.allfiles
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频文件！")
            return

        self.yuanwuyuan = 1
        self.border_label.close()
        if self.flag > 0:
            self.clearImages()
        self.flag += 1

        self.frame_counter = 0
        self.frame_number = 0
        self.frame_fake = 0
        self.frame_ture = 0

        self.timer_else.start(25)  # 30 ms 间隔, 大约33帧每秒
        self.timer_video.start(10)  # 改为每秒更新一次
        self.timer.start(25)

    def handle_error(self):
        print("Error: ", self.player1.errorString())

    def update_frame_video_else(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # 将帧调整为与 QLabel 相同大小
        frame = cv2.resize(frame, (self.video_label.width(), self.video_label.height()))

        # 将视频帧从 BGR 转换为 RGB 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将视频帧转换为 QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 将 QImage 设置到 QLabel 上
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def on_pushButton_Add_model_23(self):
        self.ModelFlag = 0
        QMessageBox.information(self, "提示", "模型一部署完毕")

    def on_pushButton_Add_model_40(self):
        self.ModelFlag = 1
        QMessageBox.information(self, "提示", "模型二部署完毕")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()


    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if not file_path:
            QMessageBox.warning(self, "警告", "请选择一个视频文件！")
            return

        self.cap = cv2.VideoCapture(file_path)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开视频文件！")
            return

        self.yuanwuyuan = 1

        self.border_label.close()
        if self.flag > 0:
            self.clearImages()
        self.flag += 1

        self.frame_counter = 0
        self.frame_number = 0
        self.frame_fake = 0
        self.frame_ture = 0

        # 设置视频路径并启动更新帧的定时器
        self.video_path = file_path
        self.timer_else.start(25)  # 30 ms 间隔, 大约33帧每秒
        self.timer_video.start(10)  # 改为每秒更新一次
        self.timer.start(25)

    def on_pushButton_Player_clicked(self):
        self.yuanwuyuan = 1
        self.timer_else.start(25)
        self.timer_video.start(10)
        self.timer.start(25)

    def on_pushButton_Pause_clicked(self):
        self.yuanwuyuan = 0
        self.timer_else.stop()
        self.timer_video.stop()
        self.timer.stop()

    def on_pushButton_Stop_clicked(self):
        self.yuanwuyuan = 0
        self.frame_counter = 0
        self.frame_number = 0
        self.frame_fake = 0
        self.frame_ture = 0
        self.timer_else.stop()
        self.timer_video.stop()
        self.timer.stop()
        self.cap.release()
        self.video_label.clear()
        self.gif_label.hide()
        self.giftext_label.hide()
        if self.flag > 0:
            self.clearImages()
        self.flag += 1

    def on_slider_value_changed(self):
        # When the slider value changes, update the video playback position and update the time label
        position_percent = self.horizontalSlider_2.value()
        duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS) * 1000
        new_position = int(position_percent * duration / 100)  # Convert to milliseconds
        self.cap.set(cv2.CAP_PROP_POS_MSEC, new_position)

        self.update_frame_video_else()

        self.update_duration_label(new_position, duration)

    def on_slider_moved(self):
        self.timer.stop()

    def update_slider_position(self):
        # Update the slider value and time label when the timer triggers, but only if the user hasn't dragged the slider
        if not self.horizontalSlider_2.isSliderDown():
            position = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps != 0 and frame_count != 0:
                duration = frame_count / fps * 1000
                value = int(position * 100 / duration)
                self.horizontalSlider_2.setValue(value)
                self.update_duration_label(position, duration)

    def update_duration_label(self, position, duration):
        current_time_str = self.format_time(position)
        total_time_str = self.format_time(duration)

        self.label_2.setText(f'{current_time_str}/{total_time_str}')

    def format_time(self, milliseconds):
        seconds = int(milliseconds / 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        return '{:02}:{:02}:{:02}'.format(hours, minutes, seconds)


    def closeEvent(self, event):
        self.close()
        self.yuanwuyuan = 0
        event.accept()  # 关闭窗口

    def minimize(self):
        self.showMinimized()



#############################################################################################################################
#################################################  登陆界面 #######################################################################
##############################################################################################################################

class LoginWindow(QWidget):
    def __init__(self, model_loader):
        self.model_loader = model_loader
        super().__init__()
        self.setFixedSize(1200, 675)  # Enlarged window size
        self.setWindowIcon(QIcon('favicon.ico'))  # Replace 'icon.png' with your icon path
        self.setWindowTitle('伪影检测 Multiple Attention')
        self.setWindowIcon(QIcon('favicon.ico'))  # Replace 'icon.png' with your icon path
        self.initUI()
        self.setStyleSheet('''
                    LoginWindow{
                        border: 5px solid #00BFFF;  /* 添加蓝光边框效果，边框宽度为5px，颜色为蓝色 */
                    }
                ''')
        self.img_label = QLabel(self)
        pixmap = QPixmap('./background/deepfake.jpg')  # 替换为你的图片路径
        self.img_label.setPixmap(pixmap)
        self.img_label.setGeometry(0, 0, pixmap.width(), pixmap.height())
        self.img_label.lower()

    def initUI(self):

        self.title_label = QLabel('伪影检测', self)
        self.title_label.setGeometry(790, 130, 500, 100)  # Repositioned title
        self.title_label.setStyleSheet("""
            font-size: 56px;
            font-weight: bold;
            color: #00bfff;
            font-family: 优设标题黑;
        """)

        self.title_label_1 = QLabel('Multiple-Attention', self)
        self.title_label_1.setGeometry(740, 210, 500, 100)  # Repositioned title

        self.title_label_1.setStyleSheet("""
            background: rgba(85, 170, 255, 0);
            color: rgb(71, 246, 229);
            font-style: SimHei;
            font: 9pt "STLiti";
            font-size: 44px;
        """)

        # Username label and entry
        self.username_label = QLabel('用户名:', self)
        self.username_label.setGeometry(700, 330, 170, 40)  # Repositioned username label
        self.username_label.setStyleSheet("""
            font-size: 28px;
            color: #00bfff;
            font-family: "方正姚体"; /* 添加字体样式 */
            font-weight: bold; /* 加粗 */
        """)

        self.username_entry = QLineEdit(self)
        self.username_entry.setGeometry(800, 330, 300, 50)  # Resized and repositioned username entry
        self.username_entry.setStyleSheet("""
            font-size: 18px;
            background-color: #272727;
            color: #ffffff;
            font-family: "STXingkai"; /* 添加字体样式 */
            border-radius: 20px; /* 设置椭圆边框 */
            font-family: "方正姚体";
        """)
        self.username_entry.setPlaceholderText('   输入您的用户名')

        # Password label and entry
        self.password_label = QLabel('密码:', self)
        self.password_label.setGeometry(710, 400, 170, 40)  # Repositioned password label
        self.password_label.setStyleSheet("""
            font-size: 28px;
            color: #00bfff;
            font-family: "方正姚体"; /* 添加字体样式 */
            font-weight: bold; /* 加粗 */
        """)

        self.password_entry = QLineEdit(self)
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.password_entry.setGeometry(800, 400, 300, 50)  # Resized and repositioned password entry
        self.password_entry.setStyleSheet("""
            font-size: 18px;
            background-color: #272727;
            color: #ffffff;
            font-family: "STXingkai"; /* 添加字体样式 */
            border-radius: 20px; /* 设置椭圆边框 */
            font-family: "方正姚体";
        """)
        self.password_entry.setPlaceholderText('   输入您的密码')

        # Agree checkbox
        self.agree_checkbox = QCheckBox('同意协议', self)
        self.agree_checkbox.setGeometry(760, 490, 200, 30)  # Resized and repositioned checkbox
        self.agree_checkbox.setStyleSheet("""
            font-size: 20px;
            color: #00bfff;
            font-weight: bold; /* 加粗 */
            font-family:方正姚体
        """)

        from PyQt5.QtGui import QLinearGradient, QColor

        # Login button
        self.login_button = QPushButton('登录', self)
        self.login_button.setGeometry(760, 550, 295, 40)  # Resized and repositioned login button
        # 设置渐变背景
        gradient = QLinearGradient(0, 0, 0, self.login_button.height())
        gradient.setColorAt(0, QColor('#00bfff'))  # 渐变开始颜色
        gradient.setColorAt(0.66, QColor('#b3e0ff'))  # 调整了渐变点位置，颜色更浅

        self.login_button.setStyleSheet("""
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                        stop: 0 #00bfff, stop: 0.8 #005cbf); /* 渐变背景色 */
            color:  #FAFCFF; /* 文本颜色 */
            font-size: 24px;
            font-weight: bold; /* 加粗 */
            font-family: SimHei; /* 字体格式 */
            border-radius: 15px; /* 圆角半径 */
            font-family:优设标题黑
        """)
        self.login_button.clicked.connect(self.onLogin)

        # Forgot password button
        self.forgot_button = QPushButton('忘记密码', self)
        self.forgot_button.setGeometry(930, 490, 150, 30)  # Resized and repositioned forgot button
        self.forgot_button.setStyleSheet("""
            background-color: transparent; /* 设置背景透明 */
            color: #00bfff; /* 设置文本颜色为白色 */
            font-size: 20px;
            font-weight: bold; /* 加粗 */
            border-radius: 5px;
            font-family:方正姚体
        """)
        self.forgot_button.clicked.connect(self.onForgotPassword)

    def onLogin(self):
        username = self.username_entry.text()
        password = self.password_entry.text()

        if username == 'xinxianquan' and password == '123456':
            if self.agree_checkbox.isChecked():
                QMessageBox.information(self, '登录成功', '欢迎回来，{}'.format(username))
                # 跳转到第一页
                self.close()
                self.first_window = HomeWindow(self.model_loader)
                self.first_window.show()
            else:
                QMessageBox.warning(self, '登录失败', '请先同意协议')
        else:
            QMessageBox.warning(self, '登录失败', '用户名或密码错误')

    def onForgotPassword(self):
        QMessageBox.information(self, '忘记密码', '请联系管理员重置密码,一切解释权归开发者所有')

    def close_camera(self):
        # 在这里编写关闭摄像头的代码
        # 检查 self.video_cap 是否存在并且是否处于打开状态
        if hasattr(self, 'video_cap') and self.video_cap.isOpened():
            self.video_cap.release()

    def closeEvent(self, event):  # 函数名固定不可变
            event.accept()  # 关闭窗口

'''
使用说明
'''
class PDFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("操作 说明")
        self.setGeometry(100, 100, 650, 900)
        self.setWindowIcon(QIcon('favicon.ico'))  # Replace 'icon.png' with your icon path
        # 设置窗口固定大小
        self.setFixedSize(650, 900)

        # 初始化可滚动的中央部件
        scroll_area = QScrollArea()
        self.setCentralWidget(scroll_area)

        # 创建垂直布局以容纳所有页面和按钮
        self.central_widget = QWidget()
        self.vertical_layout = QVBoxLayout(self.central_widget)

        # 将中央部件添加到滚动区域
        scroll_area.setWidget(self.central_widget)
        scroll_area.setWidgetResizable(True)

        # 加载PDF内容
        self.pdf_path = "软件使用手册.pdf"
        self.load_pdf()

    def load_pdf(self):
        # 打开PDF文件
        doc = fitz.open(self.pdf_path)

        # 创建一个框架以容纳所有页面的内容
        frame = QFrame()
        frame_layout = QVBoxLayout(frame)

        # 遍历PDF中的所有页面
        for page_num in range(doc.page_count):
            # 加载页面
            page = doc.load_page(page_num)

            # 创建QTextEdit以显示页面的文本内容
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.append(f"第 {page_num + 1} 页:")
            text_edit.append(page.get_text())

            # 创建QLabel以显示页面的图像
            pixmap = page.get_pixmap()
            image = QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride, QImage.Format_RGB888)
            page_image = QLabel()
            page_image.setPixmap(QPixmap.fromImage(image))

            # 将文本和图像添加到框架布局中
            frame_layout.addWidget(text_edit)
            frame_layout.addWidget(page_image)

        # 将框架添加到主布局
        self.vertical_layout.addWidget(frame)

    def closeEvent(self, event):
        self.close()
        event.accept()  # 关闭窗口




'''
团队 介绍
'''
class CustomScrollBar(QScrollBar):
    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        self.setStyleSheet(
            "QScrollBar:vertical {"
            "    background: transparent;"
            "    width: 16px;"
            "    border: none;"
            "}"
            "QScrollBar::handle:vertical {"
            "    background: #c0c0c0;"
            "    min-height: 20px;"
            "    border-radius: 10px;"
            "}"
            "QScrollBar::add-line:vertical {"
            "    background: transparent;"
            "}"
            "QScrollBar::sub-line:vertical {"
            "    background: transparent;"
            "}"
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(192, 192, 192, 150)))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 10, 10)  # 使用圆角矩形绘制滚动条

class ImageDisplay(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("团队 介绍-名片")
        self.setFixedSize(1050, 598)
        self.setWindowIcon(QIcon('favicon.ico'))

        # 设置窗口背景透明
        self.setStyleSheet("background: transparent;")

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        # 创建一个自定义滚动条
        scroll_bar = CustomScrollBar()
        scroll_area.setVerticalScrollBar(scroll_bar)

        # 创建主窗口部件
        main_widget = QWidget()
        scroll_area.setWidget(main_widget)

        # 创建垂直布局
        layout = QVBoxLayout(main_widget)

        # 加载第一张图片
        pixmap1 = QPixmap("./images/名片正.png")  # 替换为您的图片路径
        pixmap1 = pixmap1.scaled(1012, 598)  # 自定义图片大小
        label1 = QLabel()
        label1.setPixmap(pixmap1)
        layout.addWidget(label1)

        # 加载第二张图片
        pixmap2 = QPixmap("./images/名片反.png")  # 替换为您的图片路径
        pixmap2 = pixmap2.scaled(1012, 598)  # 自定义图片大小
        label2 = QLabel()
        label2.setPixmap(pixmap2)
        layout.addWidget(label2)

        self.setCentralWidget(scroll_area)

        def closeEvent(self, event):
            self.close()
            event.accept()  # 关闭窗口

###########################################################################################################################
##########################################################################################################################
#######################################################实时！！！！################################################################

class HomeWindow(QMainWindow):

    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.predict_23 = self.model_loader.predict_main_videos_23

        super().__init__()

        self.x_values = []
        self.y_values = []
        self.yuanwuyuan = 0

        self.setFixedSize(int(2560 * 0.9), int(0.9 * 1440))  # 设置固定的窗口大小
        self.move(140, 100)
        self.setWindowIcon(QIcon('favicon.ico'))
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.frame_count = 0  # 初始化帧计数器为0
        self.setWindowTitle('实时 平台')
        self.real_count = 0
        self.total_count = 0

        self.btn_minimize = QPushButton("-", self)
        self.btn_minimize.setGeometry(2200, 0, 50, 50)
        self.btn_minimize.setStyleSheet("background-color: transparent;font-size: 12pt; border: none; color: white;font-weight: bold;")
        self.btn_minimize.clicked.connect(self.minimize)

        self.btn_close = QPushButton("×", self)
        self.btn_close.setGeometry(2250, 0, 50, 50)
        self.btn_close.setStyleSheet("background-color: transparent;font-size: 12pt; border: none; color: white;font-weight: bold;")
        self.btn_close.clicked.connect(self.close)


        self.start_camera_button = QPushButton('开启 实时捕获', self)
        self.start_camera_button.setGeometry(430, 560, 360, 80)
        self.start_camera_button.setStyleSheet(
            "background-color: transparent ; font-size: 24pt; font-weight: bold; color: white ; font-family: 优设标题黑;")
        self.start_camera_button.clicked.connect(self.start_capture)
        self.start_camera_button.raise_()

        self.checkvideo_button = QPushButton('待检测视频放入该区域', self)
        self.checkvideo_button.setGeometry(430, 460, 360, 80)
        self.checkvideo_button.setStyleSheet(
            "background-color: transparent ; font-size: 16pt; font-weight: bold; color: red ; font-family: 方正姚体;")
        self.checkvideo_button.hide()

        # 设置摄像头状态显示区域的样式
        self.cam_label = QLabel(self)
        self.cam_label.setGeometry(1280, 190, 420, 460)
        self.cam_label.setStyleSheet("background-color: transparent;border: 4px solid #2e4e7e")

        #softmax显示区域
        self.probabilities_label = QLabel(self)
        self.probabilities_label.setGeometry(1795, 420, 360, 96)
        self.probabilities_label.setText("判断 区域")
        self.probabilities_label.setAlignment(Qt.AlignCenter)  # 将文本居中显示
        self.probabilities_label.setStyleSheet("background-color: transparent;color:white;font-family: 优设标题黑")

        self.probabilities_TF_label = QLabel(self)
        self.probabilities_TF_label.setGeometry(1865, 530, 220, 60)
        self.probabilities_TF_label.setAlignment(Qt.AlignCenter)  # 将文本居中显示
        self.probabilities_TF_label.setStyleSheet("background-color: transparent;color:white;border: none")

        self.probabilities_shuzi_label = QLabel(self)
        self.probabilities_shuzi_label.setGeometry(1865, 620, 220, 60)
        self.probabilities_shuzi_label.setAlignment(Qt.AlignCenter)  # 将文本居中显示
        self.probabilities_shuzi_label.setStyleSheet("background-color: transparent;color:white;border: none")

        self.probabilities_gailv_label = QLabel(self)
        self.probabilities_gailv_label.setGeometry(1910, 870, 200, 60)
        self.probabilities_gailv_label.setAlignment(Qt.AlignCenter)  # 将文本居中显示
        self.probabilities_gailv_label.setStyleSheet("background-color: transparent;color:white;border: none")

        font_pro = QFont("方正姚体", 24, QFont.Bold)
        font_pro_else = QFont("方正姚体", 16, QFont.Bold)
        font_pro_else_1 = QFont("方正姚体", 12, QFont.Bold)
        font_pro_region = QFont("方正姚体", 18, QFont.Bold)
        self.probabilities_label.setFont(font_pro_else)
        self.probabilities_TF_label.setFont(font_pro_else)
        self.probabilities_gailv_label.setFont(font_pro_else_1)
        self.probabilities_shuzi_label.setFont(font_pro_else)

        # 创建水平布局用于容纳帧
        self.frame_layout = QHBoxLayout()

        # 创建滚动区域来容纳水平布局
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setGeometry(120, 860, 940, 320)

        # 设置滚动区域的样式表
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }"
                                       "QScrollBar:vertical { width: 8px; background: transparent; border-radius: 4px; }"
                                       "QScrollBar::handle:vertical { background: #808080; border-radius: 10px; width: 4px; }"
                                       "QScrollBar::add-line:vertical { background: transparent; }"
                                       "QScrollBar::sub-line:vertical { background: transparent; }"
                                       "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
                                       "QScrollBar:horizontal { height: 8px; background: transparent; border-radius: 4px; }"
                                       "QScrollBar::handle:horizontal { background: #808080; border-radius: 10px; height: 4px; }"
                                       "QScrollBar::add-line:horizontal { background: transparent; }"
                                       "QScrollBar::sub-line:horizontal { background: transparent; }"
                                       "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }")

        # 在滚动区域内设置水平布局
        content_widget = QWidget()
        content_widget.setLayout(self.frame_layout)

        # 设置内容小部件的样式表，使其背景透明
        content_widget.setStyleSheet("background-color: transparent;")
        self.scroll_area.setWidget(content_widget)

        # 创建一个标签用于显示图片
        self.content_widget_label = QLabel(self)
        self.content_widget_label.setGeometry(160, 880, 780, 286)

        # 添加四个区域
        self.region1 = QLabel(self)
        self.region1.setGeometry(200+300-150, 15, 180, 40)
        self.region1.setText("实时 平台")
        self.region1.setStyleSheet("background-color: transparent;color:white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region1.mousePressEvent = lambda event: self.on_region_clicked(1)

        self.region2 = QLabel(self)
        self.region2.setGeometry(300+600-150, 15, 180, 40)
        self.region2.setStyleSheet("background-color: transparent;color: white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region2.mousePressEvent = lambda event: self.on_region_clicked(2)
        self.region2.setText("视频 检测")

        self.region3 = QLabel(self)
        self.region3.setGeometry(400+900-150, 15, 180, 40)
        self.region3.setStyleSheet("background-color: transparent;color: white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region3.mousePressEvent = lambda event: self.on_region_clicked(3)
        self.region3.setText("使用 说明")

        self.region4 = QLabel(self)
        self.region4.setGeometry(500+1200-150, 15, 180, 40)
        self.region4.setStyleSheet("background-color: transparent;color: white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region4.mousePressEvent = lambda event: self.on_region_clicked(4)
        self.region4.setText("团队 介绍")

        self.region5 = QLabel(self)
        self.region5.setGeometry(600+1500-150, 15, 180, 40)
        self.region5.setStyleSheet("background-color: transparent;color: white;font-family:优设标题黑")  # 例子样式，根据需要更改
        self.region5.mousePressEvent = lambda event: self.on_region_clicked(5)
        self.region5.setText("工具")

        self.region1.setFont(font_pro_region)
        self.region2.setFont(font_pro_region)
        self.region3.setFont(font_pro_region)
        self.region4.setFont(font_pro_region)
        self.region5.setFont(font_pro_region)

        #给每个区域设置蒙版
        opacity_effect1 = QGraphicsOpacityEffect()
        opacity_effect1.setOpacity(1.00)

        opacity_effect2 = QGraphicsOpacityEffect()
        opacity_effect2.setOpacity(0.33)

        opacity_effect3 = QGraphicsOpacityEffect()
        opacity_effect3.setOpacity(0.33)

        opacity_effect4 = QGraphicsOpacityEffect()
        opacity_effect4.setOpacity(0.33)

        opacity_effect5 = QGraphicsOpacityEffect()
        opacity_effect5.setOpacity(0.33)

        self.region1.setGraphicsEffect(opacity_effect1)
        self.region2.setGraphicsEffect(opacity_effect2)
        self.region3.setGraphicsEffect(opacity_effect3)
        self.region4.setGraphicsEffect(opacity_effect4)
        self.region5.setGraphicsEffect(opacity_effect5)

        # 加载图片
        pixmap_content_widget = QPixmap('./images/content_widget.png')
        self.content_widget_label.setPixmap(pixmap_content_widget)
        self.content_widget_label.setScaledContents(True)

        # 设置退出按钮的样式
        self.quit_button = QPushButton('退 出', self)
        self.quit_button.setGeometry(1920, 325, 160, 90)
        self.quit_button.setStyleSheet(
            "background-color: transparent; color: white;font-family: 优设标题黑")
        self.quit_button.clicked.connect(self.close_capture)

        # 创建一个按钮来清除图像
        self.clear_button = QPushButton("重 置", self)
        self.clear_button.setGeometry(1920, 160, 160, 90)  # 设置按钮位置和大小
        self.clear_button.setStyleSheet(
            "background-color: transparent; color: white;font-family: 优设标题黑")
        self.clear_button.clicked.connect(self.clearImages)  # 按钮点击连接到清除函数
        self.clear_button.setFont(font_pro)
        self.quit_button.setFont(font_pro)


        # 创建 accuracy折线图Label
        self.line_chart_label = QLabel(self)
        self.line_chart_label.setGeometry(1380, 930, 520, 260)
        self.line_chart_label.setStyleSheet(
            "background-color: transparent; "
            "border:none;"
        )


        # 创建一个标签用于显示 GIF 图像
        self.gif_label = QLabel(self)
        self.gif_label.setGeometry(1350, 650, 200, 120)
        # 创建一个标签用于显示 GIF 图像
        self.giftext_label = QLabel(self)
        self.giftext_label.setGeometry(1520, 650, 200, 120)
        self.giftext_label.setText("检测中")
        self.giftext_label.setStyleSheet(
            "background-color: transparent ; font-size: 14pt; font-weight: bold; color: white ; font-family: 优设标题黑;")

        # 加载 GIF 图像
        self.movie = QMovie('./images/5.gif')
        self.gif_label.setMovie(self.movie)
        self.gif_label.lower()

        self.gif_label.hide()
        self.giftext_label.hide()

        # 开始播放 GIF 图像
        self.movie.start()
        self.gif_label.setStyleSheet("background-color: transparent;")

        # 创建标签并插入图片
        image_label = QLabel(self)
        pixmap = QPixmap("./images/Final.png")  # 替换为您的图片路径
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)  # 缩放图片以适应标签大小
        image_label.setGeometry(0, 0, int(2560*0.9), int(0.9*1440))  # 设置标签的位置和大小
        # 将标签置于最底层
        image_label.lower()

    def on_region_clicked(self, region_number):
        # 为每个区域定义操作
        if region_number == 1:
            pass
        elif region_number == 2:
            self.close()
            self.secondwindow = SecondWindow(self.model_loader)
            self.secondwindow.show()
            pass
        elif region_number == 3:
            opacity_effect3 = QGraphicsOpacityEffect()
            opacity_effect3.setOpacity(1.00)
            self.region3.setGraphicsEffect(opacity_effect3)
            self.thirdwindow = PDFViewer()
            self.thirdwindow.show()
            pass
        elif region_number == 4:
            opacity_effect4 = QGraphicsOpacityEffect()
            opacity_effect4.setOpacity(1.00)
            self.region4.setGraphicsEffect(opacity_effect4)
            self.fourthwindow= ImageDisplay()
            self.fourthwindow.show()

            pass
        elif region_number == 5:
            opacity_effect5 = QGraphicsOpacityEffect()
            opacity_effect5.setOpacity(1.00)
            self.region5.setGraphicsEffect(opacity_effect5)
            self.fivethwindow= ToolOptionsWindow()
            self.fivethwindow.show()
            pass
    def clearImages(self):
        self.line_chart_label.clear()  # 清除 QLabel 中的文本或图像
        self.cam_label.clear()
        self.cam_label.setStyleSheet(f"border: 4px solid #2e4e7e;")
        self.probabilities_shuzi_label.clear()
        self.probabilities_TF_label.clear()
        self.probabilities_gailv_label.clear()
        self.checkvideo_button.hide()
        self.clear_scroll_area()
        self.x_values = []  # 用于存储 x 轴数据（例如帧数）
        self.y_values = []  # 用于存储 y 轴数据（例如准确度）
        self.frame_count = 0  # 初始化帧计数器为0
        self.real_count = 0
        self.total_count = 0
        self.yuanwuyuan = 0
        self.redo_camera()
        self.content_widget_label.show()
        self.start_camera_button.show()
        self.gif_label.hide()
        self.giftext_label.hide()

    # 清空滚动区域的内容
    def clear_scroll_area(self):
        content_widget = self.scroll_area.takeWidget()
        if content_widget:
            content_widget.deleteLater()

    def update_accuracy(self, frame_count, accuracy):
        # 添加新的数据点
        self.x_values.append(frame_count)
        self.y_values.append(accuracy)

        # 更新折线图
        self.plot_line_chart_ui(self.x_values, self.y_values)

    def plot_line_chart_ui(self, x_values, y_values):
        # 清空图表
        plt.clf()

        # 设置全局字体属性
        plt.rcParams['font.size'] = 30  # 设置字体大小
        plt.rcParams['axes.edgecolor'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        # 创建一个新的图表，并设置其大小
        plt.figure(figsize=(18, 10))  # 设置图像大小为 8x6 inches

        # 绘制折线图
        plt.plot(x_values, y_values, marker='o', color='white')
        plt.title("Line Chart", fontweight='bold', color='white')
        plt.xlabel("Frame Number", fontweight='bold', color='white')
        plt.ylabel("Accuracy", fontweight='bold', color='white')
        # 设置 x 和 y 轴的刻度文本颜色
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')

        # 显示图像到 QLabel
        temp_file_line = 'temp_line_plot.png'
        plt.savefig(temp_file_line, transparent=True)
        pixmap = QPixmap(temp_file_line)
        self.line_chart_label.setPixmap(pixmap)
        self.line_chart_label.setScaledContents(True)  # 图片自适应 Label 大小

    def update_frame(self):
        if self.yuanwuyuan == 1:
            screenshot = capture_screen()
            self.frame_count += 1
            screenshot_np = np.array(screenshot)
            h, w, c = screenshot_np.shape

            # 判断是否进行人脸检测每两帧检测一次
            if self.frame_count % 2 == 0:
                # 处理人脸
                begin = time.time()
                faces = main(screenshot_np)
                if faces is not None:

                    # 创建 QLabel 来显示帧图像
                    frame_label = QLabel()
                    frame_label.setFixedSize(210, 250)

                    x1 = int(0.95 * faces[0])
                    y1 = int(0.9 * faces[1])
                    h = int(1.2 * faces[3])
                    w = int(1.2 * faces[2])
                    x2 = w + x1
                    y2 = h + y1

                    if x1 < 0: x1 = 0
                    if x2 < 0: x2 = 0
                    if y1 < 0: y1 = 0
                    if y2 < 0: y2 = 0

                    face_img = screenshot_np[y1:y2, x1:x2]
                    frame_width = 380
                    frame_height = 380
                    face_img = cv2.resize(face_img, (frame_width, frame_height))
                    # 预测结果
                    label, probability = self.predict_23(face_img)
                    # 将单个概率转换为百分比
                    probability_percent = f"{probability * 100:.2f}%"
                    if self.total_count % 30 == 0:
                        if self.total_count - self.real_count >= 20:
                            self.real_count = 0
                            self.total_count = 0

                    # 更新正确率信息
                    self.total_count += 1
                    if label == "Real":
                        self.real_count += 1
                    else:
                        self.real_count += 0
                    accuracy = self.real_count / self.total_count

                    # 更新折线图
                    frame_number = len(self.x_values) + 1  # 帧数即为已记录数据点数量加一
                    self.update_accuracy(frame_number, accuracy)

                    # 根据预测结果设置字体颜色
                    color = (0, 255, 102) if label == "Real" else (204, 0, 0)

                    cv2.putText(face_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                    # 将图像从 BGR 格式转换为 RGB 格式
                    face_img_rgb = face_img
                    # 获取图像的高度、宽度和通道数
                    h, w, c = face_img_rgb.shape
                    qimg = QImage(face_img_rgb.data, w, h, w * c, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg).scaled(420, 460, Qt.IgnoreAspectRatio)
                    self.cam_label.setPixmap(pixmap)
                    cv2.putText(face_img, str(frame_number), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                    if label == "Real":
                        border_color = "rgb(0, 255, 102)"
                    else:
                        border_color = "rgb(204, 0, 0)"

                    if accuracy >= 0.5:
                        border_color_1 = "rgb(0, 255, 102)"
                    else:
                        border_color_1 = "rgb(204, 0, 0)"

                    accuracy_percentage = accuracy * 100


                    self.cam_label.setStyleSheet(f"border: 4px solid {border_color};")

                    self.probabilities_TF_label.setText(f"结果: {label}")
                    self.probabilities_TF_label.setStyleSheet(f"color:{border_color}")
                    self.probabilities_shuzi_label.setText(f"概率: {probability_percent}")
                    self.probabilities_shuzi_label.setStyleSheet(f"color:{border_color}")
                    self.probabilities_gailv_label.setText(f"真伪率: {accuracy_percentage:.2f}%")
                    self.probabilities_gailv_label.setStyleSheet(f"color:{border_color_1}")
                    pixmap_else = QPixmap.fromImage(qimg).scaled(210, 250, Qt.KeepAspectRatio)
                    self.content_widget_label.hide()

                    self.gif_label.show()
                    self.giftext_label.show()
                    if frame_number <= 20:
                        clickable_label = ClickableLabel(pixmap_else)
                        self.frame_layout.addWidget(clickable_label)

    def redo_camera(self):
        self.reset_scroll_area()

    # 重置滚动区域的内容到原始状态
    def reset_scroll_area(self):
        # 创建水平布局用于容纳帧
        self.frame_layout = QHBoxLayout()
        # 设置滚动区域的样式表
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }"
                                       "QScrollBar:vertical { width: 8px; background: transparent; border-radius: 4px; }"
                                       "QScrollBar::handle:vertical { background: #808080; border-radius: 10px; width: 4px; }"
                                       "QScrollBar::add-line:vertical { background: transparent; }"
                                       "QScrollBar::sub-line:vertical { background: transparent; }"
                                       "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }"
                                       "QScrollBar:horizontal { height: 8px; background: transparent; border-radius: 4px; }"
                                       "QScrollBar::handle:horizontal { background: #808080; border-radius: 10px; height: 4px; }"
                                       "QScrollBar::add-line:horizontal { background: transparent; }"
                                       "QScrollBar::sub-line:horizontal { background: transparent; }"
                                       "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }")

        # 在滚动区域内设置水平布局
        content_widget = QWidget()
        content_widget.setLayout(self.frame_layout)
        # 设置内容小部件的样式表，使其背景透明
        content_widget.setStyleSheet("background-color: transparent;")
        self.scroll_area.setWidget(content_widget)

    def start_capture(self):
        self.yuanwuyuan = 1
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # 当按钮被按下时启动计时器
        # 隐藏按钮
        self.start_camera_button.hide()
        self.checkvideo_button.show()

    def closeotherwindow(self, num):
        if num == 3:
            opacity_effect3 = QGraphicsOpacityEffect()
            opacity_effect3.setOpacity(0.33)
            self.region3.setGraphicsEffect(opacity_effect3)
        if num == 4:
            opacity_effect4 = QGraphicsOpacityEffect()
            opacity_effect4.setOpacity(0.33)
            self.region4.setGraphicsEffect(opacity_effect4)
        if num == 5:
            opacity_effect5 = QGraphicsOpacityEffect()
            opacity_effect5.setOpacity(0.33)
            self.region4.setGraphicsEffect(opacity_effect5)


    def close_capture(self):
        self.yuanwuyuan = 0
        self.gif_label.hide()
        self.giftext_label.hide()

    def closeEvent(self, event):
        self.close()
        self.yuanwuyuan = 0
        event.accept()  # 关闭窗口

    def minimize(self):
        self.showMinimized()


def capture_screen():
    screenshot = ImageGrab.grab((220, 130, 1150, 920))
    return screenshot


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QCheckBox, QFileDialog,
                             QPushButton, QLabel)
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


# Attention 模型的定义
class AttentionModule(nn.Module):
    def __init__(self, in_channels, attention_maps=4):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, attention_maps, kernel_size=1)
        self.bn = nn.BatchNorm2d(attention_maps)
        self.relu = nn.ReLU()

    def forward(self, x):
        attention_maps = self.conv(x)
        attention_maps = self.bn(attention_maps)
        attention_maps = self.relu(attention_maps)
        return attention_maps


class AttentionFramework(nn.Module):
    def __init__(self, in_channels, attention_maps=4):
        super(AttentionFramework, self).__init__()
        self.attention_module = AttentionModule(in_channels, attention_maps)

    def forward(self, x):
        attention_maps = self.attention_module(x)
        refined_features = x * attention_maps
        pooled_features = F.adaptive_avg_pool2d(refined_features, (1, 1))
        return pooled_features, attention_maps


# 工具选项窗口
class ToolOptionsWindow(QMainWindow):
    interfaceClosed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowIcon(QIcon('favicon.ico'))  # Replace 'icon.png' with your icon path

    def initUI(self):
        self.setWindowTitle('工具选项')
        self.resize(800, 400)  # 调整窗口大小以适应 Matplotlib 可视化

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()  # 使用水平布局
        central_widget.setLayout(layout)

        # 主内容区初始化
        self.mainContent = QWidget()  # 创建一个主内容区
        self.mainContentLayout = QVBoxLayout()
        self.mainContent.setLayout(self.mainContentLayout)
        layout.addWidget(self.mainContent, 1)  # 主内容区占据更多空间

        # 工具选项侧边栏初始化
        self.tools_groupbox = QGroupBox("工具选项")
        self.tools_layout = QVBoxLayout()
        self.tools_groupbox.setLayout(self.tools_layout)
        layout.addWidget(self.tools_groupbox, 0)  # 工具选项侧边栏

        # 添加伪影检测按钮
        self.addArtifactDetectionButton()

    def addArtifactDetectionButton(self):
        # 创建伪影检测按钮
        self.artifact_button = QPushButton("伪影检测")
        self.artifact_button.clicked.connect(self.runArtifactDetection)
        self.tools_layout.addWidget(self.artifact_button)

    def runArtifactDetection(self):
        # 弹出文件选择对话框，选择图片进行检测
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片进行伪影检测", "",
                                                   "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)")
        if file_path:
            self.detect_artifact(file_path)

    def detect_artifact(self, image_path):
        # 读取图像并转换为灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("无法读取图片")
            return

        # 处理图像
        image_resized = cv2.resize(image, (256, 256))
        input_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float() / 255.0

        # 初始化模型
        model = AttentionFramework(in_channels=1, attention_maps=4)

        # 执行前向传播，生成注意力图
        pooled_features, attention_maps = model(input_tensor)

        # 可视化注意力图
        self.visualize_attention(image_resized, attention_maps)

    def visualize_attention(self, image, attention_maps):
        # 清除之前的可视化内容
        for i in reversed(range(self.mainContentLayout.count())):
            widgetToRemove = self.mainContentLayout.itemAt(i).widget()
            if widgetToRemove is not None:
                widgetToRemove.setParent(None)

        # 使用 Matplotlib 进行可视化
        # 调整 figsize 增大图像尺寸，这里调整为 (24, 16) 可以显著增大图像显示尺寸
        figure = plt.Figure(figsize=(24, 16))
        canvas = FigureCanvas(figure)

        # 创建多个子图
        ax = figure.subplots(1, attention_maps.shape[1])

        # 将注意力图与原始图像叠加
        for i in range(attention_maps.shape[1]):
            att_map = attention_maps[0, i].detach().cpu().numpy()
            att_map_resized = cv2.resize(att_map, (image.shape[1], image.shape[0]))

            # 标准化注意力图用于可视化
            att_map_resized = (att_map_resized - att_map_resized.min()) / (
                    att_map_resized.max() - att_map_resized.min())

            # 显示原始图像和叠加的注意力图
            ax[i].imshow(image, cmap='gray')
            ax[i].imshow(att_map_resized, cmap='jet', alpha=0.5)  # 使用 jet colormap 叠加注意力
            ax[i].set_title(f'Attention Map{i + 1}', fontsize=10)  # 增大标题字体
            ax[i].axis('off')

        # 调整子图间距，防止标题重叠
        figure.subplots_adjust(wspace=0.5)  # 调整子图之间的水平间距

        # 将绘图区域添加到布局中
        self.mainContentLayout.addWidget(canvas)

    def closeEvent(self, event):
        self.close()
        self.interfaceClosed.emit(5)
        event.accept()  # 关闭窗口


class CaptureWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Detection")
        self.resize(400, 250)

        self.capture_button = QPushButton('一键 拍照', self)
        self.capture_button.setGeometry(100, 100, 200, 30)
        self.capture_button.clicked.connect(self.capture_face)

    def capture_face(self):
        # 初始化视频捕获对象
        self.cap = cv2.VideoCapture(0)

        # 循环直到检测到人脸
        while True:
            # 从摄像头捕获一帧
            ret, frame = self.cap.read()
            # 检查帧是否成功读取
            if not ret:
                QMessageBox.warning(None, '错误', '未能从摄像头捕获帧。')
                return

            # 显示帧
            cv2.imshow('frame', frame)

            # 检测帧中的人脸
            faces = main(frame)

            # 如果检测到人脸，执行操作并退出循环
            if faces is not None:
                # 提取人脸坐标
                x1 = int(0.95 * faces[0])
                y1 = int(0.9 * faces[1])
                h = int(1.2 * faces[3])
                w = int(1.2 * faces[2])
                x2 = w + x1
                y2 = h + y1

                # 确保坐标在范围内
                if x1 < 0: x1 = 0
                if x2 < 0: x2 = 0
                if y1 < 0: y1 = 0
                if y2 < 0: y2 = 0

                # 裁剪人脸区域
                face_img = frame[y1:y2, x1:x2]

                # 调整人脸图像大小
                frame_width = 380
                frame_height = 380
                face_img = cv2.resize(face_img, (frame_width, frame_height))

                # 打开文件对话框，让用户选择保存位置
                save_path, _ = QFileDialog.getSaveFileName(None, "Save Image", "",
                                                           "JPEG Image (*.jpg);;PNG Image (*.png)")

                # 检查用户是否取消了对话框
                if save_path:
                    # 保存图像
                    cv2.imwrite(save_path, face_img)
                    QMessageBox.information(None, '拍照完成', f'图片已成功保存到{save_path}')

                break  # 执行完操作后退出循环

        # 释放视频捕获对象并关闭所有窗口
        self.cap.release()
        cv2.destroyAllWindows()

    def closeEvent(self, event):
        self.close()
        self.cap.release()
        event.accept()  # 关闭窗口


class ImageDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setWindowTitle("图片 预览")
        self.setFixedSize(600, 600)

        layout = QVBoxLayout(self)

        self.image_label = QLabel(self)
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(600, 600)

        layout.addWidget(self.image_label)

        self.setLayout(layout)

        # 使 QLabel 接收右键点击事件
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_context_menu)


    def show_context_menu(self, position):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: white;  /* 背景颜色 */
                border: 1px solid #000;   /* 边框颜色 */
            }
            QMenu::item {
                font-weight: bold;        /* 字体加粗 */
                padding: 8px 12px;        /* 内边距 */
            }
            QMenu::item:selected {
                background-color: #a8d8ff; /* 选中项背景颜色 */
            }
        """)

        save_action = menu.addAction("保存")
        save_action.triggered.connect(self.save_image)
        menu.exec_(self.image_label.mapToGlobal(position))

    def save_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                   "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
                                                   options=options)
        if file_path:
            pixmap = self.image_label.pixmap()
            pixmap.save(file_path)


class ClickableLabel(QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setPixmap(QPixmap(image_path))
        self.setScaledContents(True)  # 使图片适应label大小
        self.setFixedSize(200, 200)  # 设置固定大小

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.show_image_dialog()

    def show_image_dialog(self):
        dialog = ImageDialog(self.image_path, self)
        dialog.exec_()

if __name__ == '__main__':
    panduan = torch.cuda.is_available()
    print(torch.cuda.is_available())
    if (torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'
    app = QApplication(sys.argv)
    loader = ModelLoader()
    loader.loadingCompleted.connect(loader.show_login_window)
    loader.show()
    sys.exit(app.exec_())

