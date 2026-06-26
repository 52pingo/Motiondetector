#ifndef MOTION_DETECTOR_H
#define MOTION_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <string>
#include <random>

struct MotionDetectorParams {
    // ---- ViBe ----
    int vibeN = 25;              // 样本数
    int vibeR = 25;              // 匹配半径
    int vibeMin = 2;             // 最少匹配数
    int vibePhi = 20;            // 更新概率 1/φ
    bool  useRGB = false;        // RGB 3通道模式
    double rgbRScale = 1.8;      // 每通道匹配半径 = vibeR * rgbRScale (AND逻辑)

    // ---- 颜色精修 (Phase 2) ----
    bool   colorRefine = true;   // 颜色FG可信度检查: 抑制颜色接近BG的疑似FP
    int    colorMinDist = 10;    // FG像素与BG颜色的最小L1距离 (保守默认, 彩色场景可提至25+)

    // ---- 预处理 ----
    double gaussianSigma = 0.8;
    int    gaussianKernel = 3;

    // ---- 时空融合 ----
    double alpha = 0.7;
    double beta  = 0.15;
    double gamma = 0.15;

    // ---- 阈值 ----
    double lambda = 5.0;
    double tau_min = 0.02;       // ViBe最低置信度 (原0.05, 调低无显著影响)

    // ---- LBSP 纹理双通道 (Phase 1) ----
    int    lbspT = 15;           // LBSP 位比较阈值
    int    lbspHammR = 1;        // LBSP 汉明距离 (1=严格, office场景推荐)
    double lbspWeight = 0.0;     // LBSP 权重 (默认关闭, 纹理丰富场景可开0.3)

    // ---- 逐像素自适应R (Phase 1) ----
    bool   adaptiveR = true;     // 开启逐像素自适应匹配半径
    double adaptRK = 0.5;        // R = vibeR + adaptRK * localStd
    int    adaptRMin = 15;       // 自适应R下限
    int    adaptRMax = 40;       // 自适应R上限
    int    adaptUpdateInterval = 5; // 每N帧更新localStd

    // ---- Guided Filter (替代ICM) ----
    double gfEpsilon = 12.0;     // 引导滤波正则化 ε
    int    gfRadius = 5;         // 引导滤波半径 (最佳: 5)
    double gfThreshold = 0.35;   // 软掩码二值化基阈值
    double gfEdgeBias = 0.0;     // 边缘额外阈值 (0=关闭)

    // ---- 后处理 ----
    int    minArea = 100;
    int    morphKernelSize = 3;
    int    erodeSize = 0;          // 边界腐蚀大小 (0=关闭, 1-2改善Precision但需配合二次填充)

    // ---- 空间流 ----
    int    localStdWindow = 5;
    double tau_sigma = 45.0;

    // ---- 光流 (稠密 Farneback) ----
    double flowWeight = 0.6;     // 光流在运动证据中的权重 (0=仅帧差, 1=仅光流)
    double flowScale = 0.5;      // 光流计算下采样比例 (1=全分辨率, 0.5=半分辨率)

    // ---- 阴影过滤 ----
    bool   shadowFilter = true;
    double shadowSatMax = 35.0;
    double shadowValRatio = 0.55;

    // ---- 时序滤波 ----
    bool temporalMedian = true;
    int  temporalWindow = 3;     // 时序窗口帧数
    int  temporalVotes = 2;      // 最少投票数 (≥votes/window)

    // ---- 静态像素过滤 ----
    int    staticThresh = 0;     // 帧差<此值+强纹理→强制BG (0=关闭, 5=推荐)

    // ---- 树叶/随机运动过滤 ----
    bool   flowConsistency = false; // 光流方向一致性 (有树叶场景开启)

    // ---- 时序滞后修复 (光流前推) ----
    bool   flowWarpAlign = true;    // 光流前推对齐 (消除拖影)
    double flowWarpWeight = 0.25;   // 前推mask与当前mask融合权重 (≤0.3)

    // ---- 鬼影抑制 ----
    bool   ghostSuppress = false;   // 鬼影抑制 (白墙/低纹理场景开启)
    double ghostStaticIoU = 0.85;   // 连通域自IoU阈值 (>此值=静止)
    double ghostFlowThresh = 0.8;   // 静止判定光流幅值阈值
    int    ghostMaxFrames = 5;      // 最大连续静止帧数 (5帧抑制)
    int    ghostLowTextureR = 12;   // 低纹理区加速遗忘

    // 深度学习 ONNX
    std::string onnxPath;           // ONNX 二值化头路径 (空=不使用)
    std::string refineOnnxPath;     // ONNX 精修网络路径 (空=不使用)

    // 性能优化
    int    maxDim = 0;              // 处理前缩放到 maxDim (0=不缩放, 推荐720)
    bool   resizeOutput = true;     // 缩放后是否还原到原始分辨率

    // ---- V9: 运动显著性过滤 ----
    double flowGateThresh = 0.10;   // 光流幅值门控 (<此值=强制BG, 过滤微颤/树叶, 归一化值[0,1])
    bool   temporalPersistence = false; // 时序持续性过滤 (默认关闭, 有延迟)
    int    persistenceFrames = 3;   // 连续FG帧数阈值
    int    persistenceDecay = 1;    // 每帧BG衰减量

    // ---- V9: 颜色相近修复 ----
    bool   edgeFill = true;         // 边缘引导种子填充
    int    edgeFillThresh = 8;      // 灰度差阈值 (seed vs neighbor)
    bool   frameDiffFill = true;    // 帧差互补检测
    int    frameDiffThresh = 15;    // 帧差阈值 (0-255)
    double frameDiffFgDensity = 0.3;// 周围FG密度阈值

    // ---- V9: 距离变换空洞填充 ----
    bool   distTransformFill = true; // 距离变换膨胀
    double distFillRadius = 3.0;    // 从核心向外膨胀的像素数

    // 中间特征导出 (训练 binarization head)
    bool   dumpIntermediates = false;
    std::string dumpDir;

    // 精修网络训练数据导出
    bool   dumpRefinement = false;
    std::string refinementDir;
};

class MotionDetector {
public:
    using Params = MotionDetectorParams;
    explicit MotionDetector(const std::string& path, const Params& p = Params());
    cv::Mat nextFrame();

    const cv::Mat& getOverlay()     const { return overlay; }
    const cv::Mat& getGrayFrame()   const { return grayFrame; }
    const cv::Mat& getBackground()  const { return bgDisplay; }
    const cv::Mat& getRawFrame()    const { return rawFrame; }
    const cv::Mat& getFusionMap()   const { return fusionMap; }
    cv::Mat getForeground() const;  // mask遮罩原帧 (前景提取)
    bool isInitialized()            const { return initialized; }
    int  getFrameCount()            const { return frameCount; }
    double getFPS()                 const { return fps; }
    int  getWidth()                 const { return p.maxDim>0 ? origCols : cols; }
    int  getHeight()                const { return p.maxDim>0 ? origRows : rows; }
    int  getTotalFrames()           const { return (int)cap.get(cv::CAP_PROP_FRAME_COUNT); }
    void loadONNX(const std::string& path);         // 加载二值化头模型
    void loadRefineONNX(const std::string& path);   // 加载精修网络模型

private:
    void preprocess(const cv::Mat& frame);

    // ViBe (RGB 3通道)
    void    initViBe(const std::vector<cv::Mat>& frames);
    cv::Mat computeViBeConfidence();
    void    updateViBe(const cv::Mat& mask);

    // ViBe (LBSP 纹理) — Phase 1
    void    initViBeLBSP(const std::vector<cv::Mat>& frames);
    cv::Mat computeLBSP(const cv::Mat& gray);
    cv::Mat computeViBeConfidenceLBSP(const cv::Mat& lbsp);
    void    updateViBeLBSP(const cv::Mat& lbsp, const cv::Mat& mask);

    // 空间流 (运动感知: 光流 + 帧差梯度)
    cv::Mat computeMotionNorm();     // 光流幅值 + 帧差梯度 融合
    cv::Mat computeLocalStdNorm();
    cv::Mat computeSpatialConfidence(const cv::Mat& m, const cv::Mat& s);

    // 融合
    cv::Mat fuseStreams(const cv::Mat& t, const cv::Mat& s);
    cv::Mat binaryClassify(const cv::Mat& fusion, const cv::Mat& temporal,
                           double threshold);

    // Dense CRF (Phase 1)
    cv::Mat denseCRF(const cv::Mat& binMask, const cv::Mat& fusion,
                     const cv::Mat& colorFrame);

    // 后处理
    cv::Mat shadowFilter(const cv::Mat& mask);
    cv::Mat colorRefine(const cv::Mat& mask);
    cv::Mat fillHoles(const cv::Mat& mask);
    cv::Mat edgeSnap(const cv::Mat& mask, const cv::Mat& colorFrame);
    cv::Mat temporalMedianFilter(const cv::Mat& mask);
    cv::Mat staticFilter(const cv::Mat& mask);
    cv::Mat flowConsistencyFilter(const cv::Mat& mask);
    cv::Mat flowWarpAlign(const cv::Mat& mask);       // 光流前推对齐
    cv::Mat ghostSuppressFilter(const cv::Mat& mask); // 鬼影抑制
    cv::Mat filterSmallComponents(const cv::Mat& mask, const cv::Mat& motionMap);
    void    buildOverlay(const cv::Mat& mask);

    static double median(const cv::Mat& mat);

    Params p;
    cv::VideoCapture cap;
    int rows=0, cols=0;
    int origRows=0, origCols=0;     // 原始分辨率(缩放前)

    // ViBe
    cv::Mat bgSamples;       // CV_8UC(vibeN) 灰, 或 CV_8UC(vibeN*3) RGB
    cv::Mat bgDisplay;       // CV_8UC3, 用于颜色精修

    // ViBe LBSP 纹理 — Phase 1
    cv::Mat lbspBgSamples;   // CV_16UC(vibeN)
    cv::Mat lbspFrame;       // CV_16UC1, 当前帧 LBSP 描述子

    // 自适应R — Phase 1
    cv::Mat perPixelStd;     // CV_32FC1, 每像素灰度标准差缓存
    cv::Mat cachedStdNorm;   // CV_32FC1, 局部标准差归一化缓存 (ONNX头输入)

    // 帧缓存
    cv::Mat grayFrame, rawFrame, colorFrame;  // colorFrame: 3通道模糊后
    cv::Mat previousGray;
    cv::Mat previousColor;
    cv::Mat cachedFlowX, cachedFlowY;  // 光流缓存(用于一致性检查)
    cv::Mat previousMask;              // 上一帧mask (光流对齐/鬼影追踪)
    cv::Mat ghostTimer;                // CV_32SC1, 每像素连续静止帧计数
    cv::Mat persistenceCount;          // CV_32SC1, 每像素 FG 持续帧计数 (V9)

    // 时序滤波
    std::deque<cv::Mat> maskHistory;

    // 输出
    cv::Mat currentMask, fusionMap, overlay;

    std::mt19937 rng;
    std::vector<cv::Mat> initFrames;
    int frameCount=0;
    bool initialized=false;
    double fps=30.0;

    // 深度学习模型 (ONNX)
    cv::dnn::Net dnnModel;
    bool dnnLoaded=false;
    cv::Size dnnInputSize={256,256};

    // 精修网络 (ONNX)
    cv::dnn::Net dnnRefineModel;
    bool dnnRefineLoaded=false;
};

#endif
