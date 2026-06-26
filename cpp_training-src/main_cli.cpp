#include "motion_detector.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(p, m) _mkdir(p)
#endif

static void printUsage(const char* prog)
{
    std::cout << "用法: " << prog << " <视频路径> [选项]\n"
              << "       " << prog << " camera  [选项]    (使用摄像头)\n\n"
              << "选项 (均为可选):\n"
              << "  --init N             ViBe背景样本数 (默认 25)\n"
              << "  --vibe-r N           ViBe匹配半径 (默认 25)\n"
              << "  --vibe-min N         ViBe最少匹配数 (默认 2)\n"
              << "  --vibe-phi N         ViBe更新概率 1/φ (默认 20)\n"
              << "  --alpha F            时间流融合权重 (默认 0.7)\n"
              << "  --beta F             空间流融合权重 (默认 0.15)\n"
              << "  --gamma F            交叉增强权重 (默认 0.15)\n"
              << "  --lambda F           自适应阈值倍数 (默认 5)\n"
              << "  --min-area N         最小连通域面积 (默认 100)\n"
              << "  --gf-radius N       引导滤波半径 (默认 7)\n"
              << "  --gf-eps F          引导滤波正则化 ε (默认 12)\n"
              << "  --gf-thresh F       引导滤波二值化阈值 (默认 0.35)\n"
              << "  --adapt-k F          自适应R系数 (默认 0.5)\n"
              << "  --no-adapt-r         关闭逐像素自适应R\n"
              << "  --show-fusion        显示融合置信度热力图\n"
              << "  --fps N              输出视频帧率 (默认自动检测)\n"
              << "  --output-video F     输出二值Mask视频\n"
              << "  --output-overlay F   输出叠加可视化视频(绿色标识)\n"
              << "  --output-fg F        输出前景提取视频(背景黑色, 目标原色)\n"
              << "  --output-dir D       输出目录 (默认 ./video_result)\n"
              << "  --batch              批处理模式，同时输出PNG帧序列\n"
              << "  --save-every N       每隔 N 帧保存PNG（配合 --batch，默认 50）\n\n"
              << "按键控制:\n"
              << "  ESC / q      退出\n"
              << "  SPACE        暂停/继续\n"
              << "  s            保存当前帧 Mask 与叠加图\n"
              << std::endl;
}

// 根据文件扩展名选择 fourcc
static int selectFourcc(const std::string& path)
{
    std::string ext;
    auto dot = path.rfind('.');
    if (dot != std::string::npos) ext = path.substr(dot);
    if (ext == ".mp4" || ext == ".mov" || ext == ".mkv")
        return cv::VideoWriter::fourcc('m','p','4','v');
    if (ext == ".avi")
        return cv::VideoWriter::fourcc('M','J','P','G');
    return cv::VideoWriter::fourcc('m','p','4','v');
}

// 从完整路径中提取文件名（不含扩展名）
static std::string stem(const std::string& path)
{
    auto slash = path.find_last_of("/\\");
    auto dot   = path.rfind('.');
    if (slash == std::string::npos) slash = 0; else slash++;
    if (dot == std::string::npos || dot < slash) dot = path.size();
    return path.substr(slash, dot - slash);
}

struct FrameStats {
    int frameNum;
    int fgPixels;
    double fgRatio;
    int numBlobs;
    double elapsedSec;
};

int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string videoPath = argv[1];
    if (videoPath == "--help" || videoPath == "-h") {
        printUsage(argv[0]);
        return 0;
    }
    if (videoPath == "camera" || videoPath == "cam") {
        videoPath = "";
    }

    // 解析参数
    MotionDetector::Params params;
    bool showFusion = false;
    bool batchMode = false;
    std::string outputDir = "video_result";
    int saveEvery = 50;
    std::string outputVideo, outputOverlayVideo, outputFgVideo;
    double userFps = 0.0;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--show-fusion") {
            showFusion = true;
        } else if (arg == "--rgb") {
            params.useRGB = true;
        } else if (arg == "--batch") {
            batchMode = true;
        } else if (arg == "--no-adapt-r") {
            params.adaptiveR = false;
        } else if (arg == "--flow-consistency") {
            params.flowConsistency = true;
        } else if (arg == "--flow-warp") {
            params.flowWarpAlign = true;
        } else if (arg == "--ghost-suppress") {
            params.ghostSuppress = true;
        } else if (i + 1 < argc) {
            std::string val = argv[i + 1];
            if (arg == "--init")           params.vibeN = std::stoi(val);
            else if (arg == "--sigma")      params.gaussianSigma = std::stod(val);
            else if (arg == "--vibe-r")     params.vibeR = std::stoi(val);
            else if (arg == "--vibe-min")   params.vibeMin = std::stoi(val);
            else if (arg == "--vibe-phi")   params.vibePhi = std::stoi(val);
            else if (arg == "--alpha")      params.alpha = std::stod(val);
            else if (arg == "--beta")       params.beta  = std::stod(val);
            else if (arg == "--gamma")      params.gamma = std::stod(val);
            else if (arg == "--lambda")     params.lambda = std::stod(val);
            else if (arg == "--min-area")   params.minArea = std::stoi(val);
            else if (arg == "--gf-radius")   params.gfRadius = std::stoi(val);
            else if (arg == "--gf-eps")      params.gfEpsilon = std::stod(val);
            else if (arg == "--gf-thresh")   params.gfThreshold = std::stod(val);
            else if (arg == "--gf-edge")     params.gfEdgeBias = std::stod(val);
            else if (arg == "--adapt-k")     params.adaptRK = std::stod(val);
            else if (arg == "--flow-w")     params.flowWeight = std::stod(val);
            else if (arg == "--rgb-rscale")  params.rgbRScale = std::stod(val);
            else if (arg == "--color-mindist") params.colorMinDist = std::stoi(val);
            else if (arg == "--static-thresh") params.staticThresh = std::stoi(val);
            else if (arg == "--output-dir") outputDir = val;
            else if (arg == "--save-every") saveEvery = std::stoi(val);
            else if (arg == "--fps")        userFps = std::stod(val);
            else if (arg == "--output-video")       outputVideo = val;
            else if (arg == "--output-overlay")     outputOverlayVideo = val;
            else if (arg == "--output-fg")         outputFgVideo = val;
            else {
                std::cerr << "未知参数: " << arg << std::endl;
                return 1;
            }
            i++;
        } else {
            std::cerr << "缺少值: " << arg << std::endl;
            return 1;
        }
    }

    // 创建检测器
    MotionDetector detector(videoPath, params);
    double fps = (userFps > 0) ? userFps : detector.getFPS();
    int w = detector.getWidth();
    int h = detector.getHeight();

    // 确保输出目录存在
    mkdir(outputDir.c_str(), 0755);

    // 如果没有指定完整输出路径，自动放在 outputDir 下
    std::string videoStem = videoPath.empty() ? "camera" : stem(videoPath);
    if (outputVideo.empty() && outputOverlayVideo.empty() && outputFgVideo.empty() && !batchMode) {
        // 默认：输出全部三种视频
        outputOverlayVideo = outputDir + "/" + videoStem + "_overlay.mp4";
        outputVideo        = outputDir + "/" + videoStem + "_mask.mp4";
        outputFgVideo      = outputDir + "/" + videoStem + "_fg.mp4";
    }

    std::cout << "============================================\n"
              << " 视频运动目标检测 — 时空双流方案\n"
              << "============================================\n"
              << "输入: " << (videoPath.empty() ? "camera" : videoPath) << "\n"
              << "视频尺寸: " << detector.getWidth()
              << " x " << detector.getHeight() << "\n"
              << "帧率: " << fps << " FPS\n"
              << "初始化帧数: " << params.vibeN << "\n"
              << "输出目录: " << outputDir << "\n"
              << "正在初始化背景模型...\n";

    // ================================================================
    // 视频输出模式
    // ================================================================
    bool videoMode = !outputVideo.empty() || !outputOverlayVideo.empty() || !outputFgVideo.empty();
    if (videoMode || batchMode) {
        cv::VideoWriter maskWriter, overlayWriter, fgWriter;
        if (!outputVideo.empty()) {
            bool ok = maskWriter.open(outputVideo, selectFourcc(outputVideo),
                                      fps, cv::Size(w, h), false);
            if (!ok) { std::cerr << "无法创建视频文件: " << outputVideo << "\n"; return 1; }
            std::cout << "Mask 视频: " << outputVideo << "\n";
        }
        if (!outputOverlayVideo.empty()) {
            bool ok = overlayWriter.open(outputOverlayVideo,
                                         selectFourcc(outputOverlayVideo),
                                         fps, cv::Size(w, h), true);
            if (!ok) { std::cerr << "无法创建视频文件: " << outputOverlayVideo << "\n"; return 1; }
            std::cout << "Overlay 视频: " << outputOverlayVideo << "\n";
        }
        if (!outputFgVideo.empty()) {
            bool ok = fgWriter.open(outputFgVideo, selectFourcc(outputFgVideo),
                                    fps, cv::Size(w, h), true);
            if (!ok) { std::cerr << "无法创建视频文件: " << outputFgVideo << "\n"; return 1; }
            std::cout << "Foreground 视频: " << outputFgVideo << "\n";
        }

        // 统计文件
        std::string statsPath = outputDir + "/" + videoStem + "_stats.txt";
        std::ofstream statsFile(statsPath);
        statsFile << std::fixed << std::setprecision(4);
        statsFile << "# 运动目标检测逐帧统计\n"
                  << "# 帧号  前景像素数  前景占比  连通域数  耗时(s)\n";

        std::vector<FrameStats> allStats;
        int totalPixels = w * h;
        cv::Mat mask;
        int saveCounter = 0;

        auto tStart = std::chrono::steady_clock::now();

        std::cout << "处理中..." << std::endl;
        while (true) {
            mask = detector.nextFrame();
            if (mask.empty()) break;

            if (detector.isInitialized()) {
                int fn = detector.getFrameCount();

                // 逐帧统计
                int fgPixels = cv::countNonZero(mask);
                double fgRatio = (double)fgPixels / totalPixels;

                // 连通域计数
                cv::Mat labels, stats, cents;
                int nBlobs = cv::connectedComponentsWithStats(
                    mask, labels, stats, cents, 8, CV_32S);
                int validBlobs = 0;
                for (int i = 1; i < nBlobs; i++) {
                    if (stats.at<int>(i, cv::CC_STAT_AREA) >= params.minArea)
                        validBlobs++;
                }

                auto tNow = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(tNow - tStart).count();

                allStats.push_back({fn, fgPixels, fgRatio, validBlobs, elapsed});
                statsFile << fn << " " << fgPixels << " " << fgRatio
                          << " " << validBlobs << " " << elapsed << "\n";

                // 写入视频
                if (maskWriter.isOpened())
                    maskWriter.write(mask);
                if (overlayWriter.isOpened())
                    overlayWriter.write(detector.getOverlay());
                if (fgWriter.isOpened()) {
                    // 前景提取: mask=255 → 保留原色, mask=0 → 黑色
                    cv::Mat fgFrame;
                    cv::bitwise_and(detector.getRawFrame(), detector.getRawFrame(),
                                    fgFrame, mask);
                    fgWriter.write(fgFrame);
                }

                // PNG 批处理
                if (batchMode && fn % saveEvery == 0) {
                    std::string idx = std::to_string(saveCounter++);
                    std::string idxPadded = std::string(5 - idx.size(), '0') + idx;
                    cv::imwrite(outputDir + "/mask_" + idxPadded + ".png", mask);
                    cv::imwrite(outputDir + "/overlay_" + idxPadded + ".png",
                                detector.getOverlay());
                }
                if (fn % 100 == 0) {
                    std::cout << "\r已处理: " << fn << " 帧, "
                              << validBlobs << " 个目标" << std::flush;
                }
            }
        }
        auto tEnd = std::chrono::steady_clock::now();
        double totalSec = std::chrono::duration<double>(tEnd - tStart).count();

        // 汇总
        if (!allStats.empty()) {
            double sumFg = 0, sumBlobs = 0;
            for (auto& s : allStats) {
                sumFg += s.fgRatio;
                sumBlobs += s.numBlobs;
            }
            double avgFgRatio = sumFg / allStats.size();
            double avgBlobs = sumBlobs / allStats.size();
            int totalFrames = allStats.size();

            statsFile << "\n# ===== 汇总 =====\n"
                      << "# 总帧数: " << totalFrames << "\n"
                      << "# 总耗时: " << totalSec << " s\n"
                      << "# 平均FPS: " << (totalFrames / totalSec) << "\n"
                      << "# 视频分辨率: " << w << "x" << h << "\n"
                      << "# 总像素数: " << totalPixels << "\n"
                      << "# 平均前景像素数: "
                      << (int)(avgFgRatio * totalPixels) << "\n"
                      << "# 平均前景占比: " << avgFgRatio << "\n"
                      << "# 平均检测目标数: " << avgBlobs << "\n";
            statsFile.close();

            std::cout << "\n============================================\n"
                      << "  处理完成\n"
                      << "============================================\n"
                      << "总帧数:      " << totalFrames << "\n"
                      << "总耗时:      " << totalSec << " s\n"
                      << "平均帧率:    " << (totalFrames / totalSec) << " FPS\n"
                      << "平均前景占比: " << (avgFgRatio * 100) << " %\n"
                      << "平均目标数:  " << avgBlobs << "\n"
                      << "统计文件:    " << statsPath << "\n";
        }

        if (maskWriter.isOpened())    maskWriter.release();
        if (overlayWriter.isOpened()) overlayWriter.release();
        if (fgWriter.isOpened())      fgWriter.release();

        std::cout << "输出文件:\n";
        if (!outputVideo.empty())
            std::cout << "  " << outputVideo << " (二值Mask)\n";
        if (!outputOverlayVideo.empty())
            std::cout << "  " << outputOverlayVideo << " (叠加可视化)\n";
        if (!outputFgVideo.empty())
            std::cout << "  " << outputFgVideo << " (前景提取)\n";
        if (batchMode)
            std::cout << "  " << outputDir << "/mask_*.png\n"
                      << "  " << outputDir << "/overlay_*.png\n";

        return 0;
    }

    // ================================================================
    // GUI 交互模式
    // ================================================================
    const std::string winOriginal = "Original (原帧)";
    const std::string winMask     = "Mask (二值分割)";
    const std::string winOverlay  = "Overlay (叠加)";
    const std::string winFusion   = "Fusion Heatmap (融合热力图)";

    cv::namedWindow(winOriginal, cv::WINDOW_NORMAL);
    cv::namedWindow(winMask,     cv::WINDOW_NORMAL);
    cv::namedWindow(winOverlay,  cv::WINDOW_NORMAL);
    if (showFusion) {
        cv::namedWindow(winFusion, cv::WINDOW_NORMAL);
    }

    cv::Mat mask, fusionVis;
    int saveCounter = 0;
    bool paused = false;

    while (true) {
        if (!paused) {
            mask = detector.nextFrame();
            if (mask.empty()) {
                std::cout << "视频处理完毕。\n";
                break;
            }
        }

        const cv::Mat& gray = detector.getGrayFrame();
        if (!gray.empty()) {
            cv::imshow(winOriginal, gray);
        }

        if (!mask.empty()) {
            cv::imshow(winMask, mask);
        }

        const cv::Mat& overlay = detector.getOverlay();
        if (!overlay.empty()) {
            cv::imshow(winOverlay, overlay);
        }

        if (showFusion && detector.isInitialized()) {
            const cv::Mat& fusion = detector.getFusionMap();
            if (!fusion.empty()) {
                double fMin, fMax;
                cv::minMaxLoc(fusion, &fMin, &fMax);
                cv::Mat fusionNorm;
                fusion.convertTo(fusionNorm, CV_8UC1, 255.0 / std::max(fMax, 0.01));
                cv::applyColorMap(fusionNorm, fusionVis, cv::COLORMAP_JET);
                cv::imshow(winFusion, fusionVis);
            }
        }

        int fn = detector.getFrameCount();
        std::string overlayTitle = "Overlay - Frame " + std::to_string(fn);
        if (paused) overlayTitle += " [PAUSED]";
        cv::setWindowTitle(winOverlay, overlayTitle);

        int key = cv::waitKey(paused ? 0 : 10) & 0xFF;
        switch (key) {
        case 27:   // ESC
        case 'q':
        case 'Q':
            std::cout << "用户退出。\n";
            cv::destroyAllWindows();
            return 0;
        case ' ':   // SPACE
            paused = !paused;
            break;
        case 's':
        case 'S': {
            mkdir(outputDir.c_str(), 0755);
            std::string idx = std::to_string(saveCounter++);
            cv::imwrite(outputDir + "/mask_" + idx + ".png", mask);
            cv::imwrite(outputDir + "/overlay_" + idx + ".png", overlay);
            std::cout << "已保存: " << outputDir << "/mask_" << idx
                      << ".png, overlay_" << idx << ".png\n";
            break;
        }
        default:
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
