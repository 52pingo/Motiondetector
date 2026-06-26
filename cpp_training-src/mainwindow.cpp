#include "mainwindow.h"
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QFileDialog>
#include <QFileInfo>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QMessageBox>
#include <QDir>
#include <QPixmap>
#include <QGroupBox>
#include <QFormLayout>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QHeaderView>
#include <QTableWidgetItem>
#include <QAbstractItemView>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setupUi();
    setupStyle();

    connect(chooseButton, &QPushButton::clicked, this, &MainWindow::chooseVideo);
    connect(startButton, &QPushButton::clicked, this, &MainWindow::startDetection);
    connect(playButton, &QPushButton::clicked, this, &MainWindow::playResult);
    connect(stopButton, &QPushButton::clicked, this, &MainWindow::stopDetection);
    connect(exportButton, &QPushButton::clicked, this, &MainWindow::exportFgVideo);
    connect(&timer, &QTimer::timeout, this, &MainWindow::processNextFrame);
}

MainWindow::~MainWindow()
{
    timer.stop();
}

void MainWindow::setupUi()
{
    auto *central = new QWidget(this);
    setCentralWidget(central);

    auto *leftPanel = new QWidget(this);
    auto *centerPanel = new QWidget(this);
    auto *rightPanel = new QWidget(this);

    leftPanel->setFixedWidth(300);
    rightPanel->setFixedWidth(320);

    videoPathEdit = new QLineEdit(this);
    videoPathEdit->setPlaceholderText("请选择 MP4 文件");

    chooseButton = new QPushButton("选择视频", this);
    startButton  = new QPushButton("开始检测", this);
    playButton   = new QPushButton("播放结果", this);
    stopButton   = new QPushButton("停止", this);
    exportButton = new QPushButton("导出 FG 视频", this);

    startButton->setObjectName("startButton");
    stopButton->setObjectName("stopButton");
    playButton->setObjectName("playButton");
    exportButton->setObjectName("exportButton");

    vibeNSpin = new QSpinBox(this);
    vibeNSpin->setRange(5, 200);  vibeNSpin->setValue(25);

    vibeRSpin = new QSpinBox(this);
    vibeRSpin->setRange(1, 100);  vibeRSpin->setValue(25);

    minAreaSpin = new QSpinBox(this);
    minAreaSpin->setRange(1, 10000);  minAreaSpin->setValue(100);

    alphaSpin = new QDoubleSpinBox(this);
    alphaSpin->setRange(0.0, 1.0); alphaSpin->setSingleStep(0.05);
    alphaSpin->setValue(0.7);

    lambdaSpin = new QDoubleSpinBox(this);
    lambdaSpin->setRange(0.1, 20.0); lambdaSpin->setSingleStep(0.1);
    lambdaSpin->setValue(5.0);

    gfRadiusSpin = new QSpinBox(this);
    gfRadiusSpin->setRange(1, 15);  gfRadiusSpin->setValue(5);
    gfRadiusSpin->setToolTip("引导滤波半径(越小边界越锐)");

    gfThreshSpin = new QDoubleSpinBox(this);
    gfThreshSpin->setRange(0.1, 0.8); gfThreshSpin->setSingleStep(0.02);
    gfThreshSpin->setValue(0.35);
    gfThreshSpin->setToolTip("二值化阈值(越高精度↑召回↓)");

    flowWSpin = new QDoubleSpinBox(this);
    flowWSpin->setRange(0.0, 1.0); flowWSpin->setSingleStep(0.1);
    flowWSpin->setValue(0.6);
    flowWSpin->setToolTip("光流权重(0=仅帧差, 1=仅光流)");

    progressBar = new QProgressBar(this);
    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setFormat("%p%");

    fgTitleLabel = new QLabel("FG 视频结果", this);
    maskTitleLabel = new QLabel("Mask 前景分割结果", this);

    fgLabel = new QLabel(this);
    maskLabel = new QLabel(this);
    fgLabel->setMinimumSize(520, 320);
    maskLabel->setMinimumSize(520, 320);
    fgLabel->setMaximumHeight(320);
    maskLabel->setMaximumHeight(320);
    fgLabel->setAlignment(Qt::AlignCenter);
    maskLabel->setAlignment(Qt::AlignCenter);
    fgLabel->setText("等待处理完成后播放");
    maskLabel->setText("等待处理完成后播放");

    frameValueLabel = new QLabel("0", this);
    fgPixelsValueLabel = new QLabel("0", this);
    fgRatioValueLabel = new QLabel("0.00%", this);
    blobCountValueLabel = new QLabel("0", this);
    statusValueLabel = new QLabel("未开始", this);

    logOutput = new QTextEdit(this);
    logOutput->setReadOnly(true);
    logOutput->setMinimumHeight(220);

    resultTable = new QTableWidget(this);
    resultTable->setColumnCount(4);
    resultTable->setHorizontalHeaderLabels(QStringList() << "编号" << "面积" << "X" << "Y");
    resultTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    resultTable->horizontalHeader()->setStretchLastSection(true);
    resultTable->verticalHeader()->setVisible(false);
    resultTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    resultTable->setSelectionMode(QAbstractItemView::NoSelection);
    resultTable->setMinimumHeight(220);

    auto *fileGroup = new QGroupBox("文件选择", this);
    auto *fileLayout = new QVBoxLayout;
    fileLayout->addWidget(new QLabel("视频文件路径", this));
    fileLayout->addWidget(videoPathEdit);
    fileLayout->addWidget(chooseButton);
    fileGroup->setLayout(fileLayout);

    auto *paramForm = new QFormLayout;
    paramForm->addRow("ViBe 样本数", vibeNSpin);
    paramForm->addRow("ViBe 半径", vibeRSpin);
    paramForm->addRow("最小面积", minAreaSpin);
    paramForm->addRow("时间流权重 α", alphaSpin);
    paramForm->addRow("阈值倍数 λ", lambdaSpin);
    paramForm->addRow("GF 半径", gfRadiusSpin);
    paramForm->addRow("GF 阈值", gfThreshSpin);
    paramForm->addRow("光流权重", flowWSpin);

    auto *controlGroup = new QGroupBox("参数设置", this);
    controlGroup->setLayout(paramForm);

    auto *progressGroup = new QGroupBox("处理进度", this);
    auto *progressLayout = new QVBoxLayout;
    progressLayout->addWidget(progressBar);
    progressGroup->setLayout(progressLayout);

    auto *buttonGroup = new QGroupBox("操作控制", this);
    auto *buttonLayout = new QVBoxLayout;
    buttonLayout->addWidget(startButton);
    buttonLayout->addWidget(playButton);
    buttonLayout->addWidget(stopButton);
    buttonLayout->addWidget(exportButton);
    buttonGroup->setLayout(buttonLayout);

    auto *leftLayout = new QVBoxLayout;
    leftLayout->addWidget(fileGroup);
    leftLayout->addWidget(controlGroup);
    leftLayout->addWidget(progressGroup);
    leftLayout->addWidget(buttonGroup);
    leftLayout->addStretch();
    leftPanel->setLayout(leftLayout);

    auto *fgGroup = new QGroupBox("FG 可视化视频", this);
    auto *fgLayout = new QVBoxLayout;
    fgLayout->addWidget(fgTitleLabel);
    fgLayout->addWidget(fgLabel);
    fgGroup->setLayout(fgLayout);

    auto *maskGroup = new QGroupBox("前景分割图", this);
    auto *maskLayout = new QVBoxLayout;
    maskLayout->addWidget(maskTitleLabel);
    maskLayout->addWidget(maskLabel);
    maskGroup->setLayout(maskLayout);

    auto *centerLayout = new QVBoxLayout;
    centerLayout->addWidget(fgGroup);
    centerLayout->addWidget(maskGroup);
    centerLayout->addStretch();
    centerPanel->setLayout(centerLayout);

    auto *statsForm = new QFormLayout;
    statsForm->addRow("当前帧:", frameValueLabel);
    statsForm->addRow("前景像素:", fgPixelsValueLabel);
    statsForm->addRow("前景占比:", fgRatioValueLabel);
    statsForm->addRow("目标数量:", blobCountValueLabel);
    statsForm->addRow("运行状态:", statusValueLabel);

    auto *statsGroup = new QGroupBox("实时统计", this);
    statsGroup->setLayout(statsForm);

    auto *tableGroup = new QGroupBox("检测结果", this);
    auto *tableLayout = new QVBoxLayout;
    tableLayout->addWidget(resultTable);
    tableGroup->setLayout(tableLayout);

    auto *logGroup = new QGroupBox("运行日志", this);
    auto *logLayout = new QVBoxLayout;
    logLayout->addWidget(logOutput);
    logGroup->setLayout(logLayout);

    auto *rightLayout = new QVBoxLayout;
    rightLayout->addWidget(statsGroup);
    rightLayout->addWidget(tableGroup);
    rightLayout->addWidget(logGroup);
    rightPanel->setLayout(rightLayout);

    auto *mainLayout = new QHBoxLayout;
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(12);
    mainLayout->addWidget(leftPanel);
    mainLayout->addWidget(centerPanel, 1);
    mainLayout->addWidget(rightPanel);

    central->setLayout(mainLayout);

    setWindowTitle("运动目标检测可视化系统");
    resize(1600, 900);
}

void MainWindow::setupStyle()
{
    setStyleSheet(
        // ── 基础色板 (GitHub Dark 风格) ──
        "QMainWindow { background-color: #0D1117; }"
        "* { font-family: \"Segoe UI\", \"Microsoft YaHei\", sans-serif; }"

        // ── 通用控件 ──
        "QLabel { color: #E6EDF3; font-size: 13px; }"
        "QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QTableWidget, QProgressBar {"
        "  background-color: #0D1117;"
        "  color: #E6EDF3;"
        "  font-size: 13px;"
        "  border: 1px solid #30363D;"
        "  border-radius: 6px;"
        "  padding: 6px 10px;"
        "  selection-background-color: #1F6FEB;"
        "}"
        "QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {"
        "  border: 1px solid #58A6FF;"
        "}"

        // ── 按钮 ──
        "QPushButton {"
        "  background-color: #238636;"
        "  color: #FFFFFF;"
        "  border: 1px solid rgba(240,246,252,0.1);"
        "  border-radius: 6px;"
        "  padding: 8px 16px;"
        "  font-size: 13px;"
        "  font-weight: 600;"
        "  min-height: 34px;"
        "}"
        "QPushButton:hover { background-color: #2EA043; }"
        "QPushButton:pressed { background-color: #196C2E; }"
        "QPushButton#startButton { background-color: #1F6FEB; }"
        "QPushButton#startButton:hover { background-color: #388BFD; }"
        "QPushButton#startButton:pressed { background-color: #1158C7; }"
        "QPushButton#stopButton { background-color: #DA3633; }"
        "QPushButton#stopButton:hover { background-color: #F85149; }"
        "QPushButton#stopButton:pressed { background-color: #B62324; }"
        "QPushButton#exportButton { background-color: #7C3AED; }"
        "QPushButton#exportButton:hover { background-color: #8B5CF6; }"
        "QPushButton#playButton { background-color: #238636; }"
        "QPushButton#playButton:hover { background-color: #2EA043; }"

        // ── 分组框 ──
        "QGroupBox {"
        "  background-color: #161B22;"
        "  border: 1px solid #30363D;"
        "  border-radius: 10px;"
        "  margin-top: 14px;"
        "  padding: 16px 12px 12px 12px;"
        "  font-size: 13px;"
        "  font-weight: 600;"
        "  color: #E6EDF3;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 12px;"
        "  padding: 0 6px;"
        "  color: #58A6FF;"
        "}"

        // ── 表格 ──
        "QTableWidget {"
        "  background-color: #0D1117;"
        "  gridline-color: #21262D;"
        "  border: 1px solid #30363D;"
        "}"
        "QTableWidget::item { padding: 4px 8px; }"
        "QHeaderView::section {"
        "  background-color: #161B22;"
        "  color: #E6EDF3;"
        "  border: none;"
        "  border-bottom: 1px solid #30363D;"
        "  padding: 8px;"
        "  font-weight: 600;"
        "}"

        // ── 进度条 ──
        "QProgressBar {"
        "  background-color: #0D1117;"
        "  border: 1px solid #30363D;"
        "  border-radius: 4px;"
        "  text-align: center;"
        "  color: #E6EDF3;"
        "  font-size: 12px;"
        "  height: 20px;"
        "}"
        "QProgressBar::chunk {"
        "  background-color: #1F6FEB;"
        "  border-radius: 3px;"
        "}"

        // ── 滚动条 ──
        "QScrollBar:vertical {"
        "  background: #0D1117;"
        "  width: 8px;"
        "  border-radius: 4px;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #30363D;"
        "  border-radius: 4px;"
        "  min-height: 30px;"
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"

        // ── 文字编辑器 ──
        "QTextEdit {"
        "  background-color: #0D1117;"
        "  color: #8B949E;"
        "  font-size: 12px;"
        "  font-family: \"Cascadia Code\", \"Consolas\", monospace;"
        "}"
    );

    // 图片标签通透
    fgLabel->setStyleSheet(
        "background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px;");
    maskLabel->setStyleSheet(
        "background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px;");
    fgTitleLabel->setStyleSheet(
        "font-size: 15px; font-weight: 700; color: #E6EDF3; padding: 4px 0;");
    maskTitleLabel->setStyleSheet(
        "font-size: 15px; font-weight: 700; color: #E6EDF3; padding: 4px 0;");

    // 统计值高亮
    frameValueLabel->setStyleSheet("color: #58A6FF; font-weight: 600;");
    fgPixelsValueLabel->setStyleSheet("color: #F0883E; font-weight: 600;");
    fgRatioValueLabel->setStyleSheet("color: #3FB950; font-weight: 600;");
    blobCountValueLabel->setStyleSheet("color: #A371F7; font-weight: 600;");
    statusValueLabel->setStyleSheet("color: #E6EDF3; font-weight: 600;");
}

void MainWindow::chooseVideo()
{
    QString desktopPath = QDir::homePath() + "/Desktop";
    QString filePath = QFileDialog::getOpenFileName(
        this,
        "选择 MP4 视频",
        desktopPath,
        "Video Files (*.mp4 *.MP4)"
    );

    if (!filePath.isEmpty()) {
        videoPathEdit->setText(filePath);
        logMessage("已选择视频: " + filePath);
        statusValueLabel->setText("已加载文件");
    }
}

void MainWindow::startDetection()
{
    QString path = videoPathEdit->text().trimmed();
    if (path.isEmpty()) {
        QMessageBox::warning(this, "提示", "请先选择一个 MP4 文件");
        return;
    }
    if (!QFileInfo::exists(path)) {
        QMessageBox::warning(this, "错误", "视频文件不存在:\n" + path);
        return;
    }

    resetDetector();
    startButton->setText("处理中...");
    startButton->setEnabled(false);
    statusValueLabel->setText("处理中");
    logMessage("开始处理视频: " + path);

    // 快照参数(播放阶段不变)
    snapshotMinArea = minAreaSpin->value();

    try {
        MotionDetector::Params params;
        params.vibeN = vibeNSpin->value();
        params.vibeR = vibeRSpin->value();
        params.minArea = snapshotMinArea;
        params.alpha = alphaSpin->value();
        params.lambda = lambdaSpin->value();
        params.gfRadius = gfRadiusSpin->value();
        params.gfThreshold = gfThreshSpin->value();
        params.flowWeight = flowWSpin->value();

        detector = std::make_unique<MotionDetector>(path.toStdString(), params);

        double fps = detector->getFPS();
        playbackInterval = (fps > 1.0 && fps < 120.0) ? static_cast<int>(1000.0/fps) : 60;

        int totalFrames = detector->getTotalFrames();
        if (totalFrames <= 0) totalFrames = 1;
        progressBar->setRange(0, 100);

        while (true) {
            cv::Mat mask = detector->nextFrame();
            if (mask.empty()) break;

            maskFrames.push_back(mask.clone());

            cv::Mat fgFrame = detector->getForeground();
            if (!fgFrame.empty()) {
                fgFrames.push_back(fgFrame.clone());
            } else {
                fgFrames.push_back(cv::Mat::zeros(mask.size(), CV_8UC3));
            }

            int current = static_cast<int>(maskFrames.size());
            int percent = std::min(100, current * 100 / totalFrames);
            progressBar->setValue(percent);

            if (current % 50 == 0 || current == 1) {
                logMessage(QString("已处理 %1 / %2 帧 (%3%)")
                    .arg(current).arg(totalFrames).arg(percent));
            }

            qApp->processEvents();
        }

        int syncedFrames = std::min(maskFrames.size(), fgFrames.size());
        if (syncedFrames <= 0) {
            processingFinished = false;
            startButton->setText("开始检测");
            startButton->setEnabled(true);
            statusValueLabel->setText("无结果");
            logMessage("FG 视频或 Mask 结果为空");
            QMessageBox::information(this, "提示", "FG 视频或 Mask 结果为空，无法播放");
            return;
        }

        maskFrames.resize(syncedFrames);
        fgFrames.resize(syncedFrames);

        processingFinished = true;
        playbackIndex = 0;
        progressBar->setValue(100);

        startButton->setText("开始检测");
        startButton->setEnabled(true);
        statusValueLabel->setText("处理完成");
        logMessage(QString("处理完成，共 %1 帧 (初始%2帧)").arg(syncedFrames).arg(params.vibeN));

        showImage(fgLabel, fgFrames.front());
        showImage(maskLabel, maskFrames.front());
        updateStats(maskFrames.front());
    } catch (const std::exception& e) {
        startButton->setText("开始检测");
        startButton->setEnabled(true);
        QMessageBox::critical(this, "错误", QString("处理失败:\n%1").arg(e.what()));
        statusValueLabel->setText("处理失败");
        logMessage(QString("处理失败: %1").arg(e.what()));
    }
}

void MainWindow::playResult()
{
    if (!processingFinished || fgFrames.empty() || maskFrames.empty()) {
        QMessageBox::information(this, "提示", "请先完成检测处理，再播放结果");
        return;
    }

    timer.stop();
    showImage(fgLabel, fgFrames.front());
    showImage(maskLabel, maskFrames.front());
    updateStats(maskFrames.front());

    playbackIndex = 1;
    isPlaying = true;
    statusValueLabel->setText("播放中");
    logMessage("开始播放结果");
    timer.start(playbackInterval);
}

void MainWindow::stopDetection()
{
    timer.stop();
    isPlaying = false;
    statusValueLabel->setText("已停止");
    logMessage("已停止");
}

void MainWindow::exportFgVideo()
{
    if (!processingFinished || fgFrames.empty()) {
        QMessageBox::information(this, "提示", "请先完成检测处理，再导出视频");
        return;
    }

    QString savePath = QFileDialog::getSaveFileName(
        this,
        "保存 FG 视频",
        QDir::homePath() + "/Desktop/output_fg.mp4",
        "MP4 Video (*.mp4)"
    );

    if (savePath.isEmpty()) {
        return;
    }

    double fps = playbackInterval > 0 ? 1000.0 / playbackInterval : 30.0;
    cv::VideoWriter writer;
    bool ok = writer.open(
        savePath.toStdString(),
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(fgFrames.front().cols, fgFrames.front().rows),
        true
    );

    if (!ok) {
        QMessageBox::critical(this, "错误", "无法创建输出视频文件");
        return;
    }

    for (const auto& frame : fgFrames) {
        writer.write(frame);
    }
    writer.release();

    exportPath = savePath;
    logMessage("导出完成: " + savePath);
    QMessageBox::information(this, "提示", "FG 视频导出完成");
}

void MainWindow::processNextFrame()
{
    if (!isPlaying || playbackIndex >= static_cast<int>(fgFrames.size())) {
        timer.stop();
        isPlaying = false;
        if (processingFinished) {
            statusValueLabel->setText("播放完成");
            logMessage("结果播放完成");
        }
        return;
    }

    showImage(fgLabel, fgFrames[playbackIndex]);
    showImage(maskLabel, maskFrames[playbackIndex]);
    updateStats(maskFrames[playbackIndex]);
    playbackIndex++;
}

void MainWindow::updateStats(const cv::Mat& mask)
{
    if (mask.empty()) {
        return;
    }

    int frameNum = playbackIndex + 1;
    int fgPixels = cv::countNonZero(mask);
    int totalPixels = mask.rows * mask.cols;
    double fgRatio = totalPixels > 0 ? (100.0 * fgPixels / totalPixels) : 0.0;

    cv::Mat labels, stats, centroids;
    int components = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    frameValueLabel->setText(QString::number(frameNum));
    fgPixelsValueLabel->setText(QString::number(fgPixels));
    fgRatioValueLabel->setText(QString::number(fgRatio, 'f', 2) + "%");

    resultTable->setRowCount(0);
    int validCount = 0;

    for (int i = 1; i < components; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < snapshotMinArea) {
            continue;
        }

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);

        int row = resultTable->rowCount();
        resultTable->insertRow(row);
        resultTable->setItem(row, 0, new QTableWidgetItem(QString::number(validCount + 1)));
        resultTable->setItem(row, 1, new QTableWidgetItem(QString::number(area)));
        resultTable->setItem(row, 2, new QTableWidgetItem(QString::number(x)));
        resultTable->setItem(row, 3, new QTableWidgetItem(QString::number(y)));

        validCount++;
    }

    blobCountValueLabel->setText(QString::number(validCount));
}

void MainWindow::logMessage(const QString& text)
{
    logOutput->append(text);
}

void MainWindow::showImage(QLabel* label, const cv::Mat& mat)
{
    QImage image = matToQImage(mat);
    if (image.isNull()) {
        return;
    }

    label->setText("");

    QPixmap pixmap = QPixmap::fromImage(image);
    label->setPixmap(
        pixmap.scaled(label->size(), Qt::KeepAspectRatio, Qt::FastTransformation)
    );
}

QImage MainWindow::matToQImage(const cv::Mat& mat)
{
    if (mat.empty()) {
        return QImage();
    }

    if (mat.type() == CV_8UC1) {
        return QImage(
            mat.data,
            mat.cols,
            mat.rows,
            static_cast<int>(mat.step),
            QImage::Format_Grayscale8
        ).copy();
    }

    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(
            rgb.data,
            rgb.cols,
            rgb.rows,
            static_cast<int>(rgb.step),
            QImage::Format_RGB888
        ).copy();
    }

    return QImage();
}

void MainWindow::resetDetector()
{
    timer.stop();
    detector.reset();

    fgFrames.clear();
    maskFrames.clear();
    playbackIndex = 0;
    playbackInterval = 60;
    processingFinished = false;
    isPlaying = false;
    exportPath.clear();

    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    frameValueLabel->setText("0");
    fgPixelsValueLabel->setText("0");
    fgRatioValueLabel->setText("0.00%");
    blobCountValueLabel->setText("0");
    statusValueLabel->setText("未开始");
    resultTable->setRowCount(0);

    fgLabel->setPixmap(QPixmap());
    fgLabel->setText("等待处理完成后播放");
    maskLabel->setPixmap(QPixmap());
    maskLabel->setText("等待处理完成后播放");
}
