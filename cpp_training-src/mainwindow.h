#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QTableWidget>
#include <QProgressBar>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "motion_detector.h"

QT_BEGIN_NAMESPACE
class QLabel;
class QPushButton;
class QLineEdit;
class QTextEdit;
class QSpinBox;
class QDoubleSpinBox;
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void chooseVideo();
    void startDetection();
    void playResult();
    void stopDetection();
    void exportFgVideo();
    void processNextFrame();

private:
    void setupUi();
    void setupStyle();
    void resetDetector();
    void updateStats(const cv::Mat& mask);
    void logMessage(const QString& text);
    void showImage(QLabel* label, const cv::Mat& mat);
    QImage matToQImage(const cv::Mat& mat);

private:
    QLineEdit* videoPathEdit;
    QPushButton* chooseButton;
    QPushButton* startButton;
    QPushButton* playButton;
    QPushButton* stopButton;
    QPushButton* exportButton;

    QSpinBox* vibeNSpin;
    QSpinBox* vibeRSpin;
    QSpinBox* minAreaSpin;
    QDoubleSpinBox* alphaSpin;
    QDoubleSpinBox* lambdaSpin;
    QSpinBox* gfRadiusSpin;
    QDoubleSpinBox* gfThreshSpin;
    QDoubleSpinBox* flowWSpin;

    QProgressBar* progressBar;

    QLabel* fgTitleLabel;
    QLabel* maskTitleLabel;
    QLabel* fgLabel;
    QLabel* maskLabel;

    QLabel* frameValueLabel;
    QLabel* fgPixelsValueLabel;
    QLabel* fgRatioValueLabel;
    QLabel* blobCountValueLabel;
    QLabel* statusValueLabel;

    QTextEdit* logOutput;
    QTableWidget* resultTable;

    QTimer timer;
    std::unique_ptr<MotionDetector> detector;

    std::vector<cv::Mat> fgFrames;
    std::vector<cv::Mat> maskFrames;
    int playbackIndex = 0;
    int playbackInterval = 60;
    int snapshotMinArea = 100;
    bool processingFinished = false;
    bool isPlaying = false;

    QString exportPath;
};

#endif
