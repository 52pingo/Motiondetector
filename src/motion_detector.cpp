#include "motion_detector.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <cstring>

MotionDetector::MotionDetector(const std::string& path, const Params& params)
    : p(params), rng(42)
{
    if(path.empty()) cap.open(0); else cap.open(path);
    if(!cap.isOpened()) throw std::runtime_error("Cannot open: "+path);
    rows=(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cols=(int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    origRows=rows; origCols=cols;
    fps=cap.get(cv::CAP_PROP_FPS);
    if(fps<=0||fps>120) fps=30.0;
    // 图像序列可能返回0：读取首帧获取实际尺寸后重新打开
    if(rows==0||cols==0){
        cv::Mat first; cap.read(first);
        if(!first.empty()){ rows=first.rows; cols=first.cols; origRows=rows; origCols=cols; }
        cap.release(); cap.open(path);
    }
    // 处理分辨率限制：如果 maxDim>0 且原分辨率超出，缩放到 maxDim
    if(p.maxDim>0 && (rows>p.maxDim || cols>p.maxDim)){
        double s=(double)p.maxDim/std::max(rows,cols);
        rows=(int)(rows*s); cols=(int)(cols*s);
        std::cout<<"Resolution: "<<origCols<<"x"<<origRows
                 <<" -> "<<cols<<"x"<<rows<<" (maxDim="<<p.maxDim<<")\n";
    }
    if(p.gaussianKernel%2==0)p.gaussianKernel++;
    if(p.morphKernelSize%2==0)p.morphKernelSize++;
    initFrames.reserve(p.vibeN);
    // 加载 ONNX 深度学习模型 (如果指定)
    //if(!p.onnxPath.empty()) loadONNX(p.onnxPath);
}

void MotionDetector::loadONNX(const std::string& path){
    try {
        dnnModel=cv::dnn::readNetFromONNX(path);
        dnnLoaded=true;
        std::cout << "ONNX model loaded: " << path << "\n";
    } catch(const cv::Exception& e){
        std::cerr << "Failed to load ONNX: " << e.what() << "\n";
        dnnLoaded=false;
    }
}

void MotionDetector::loadRefineONNX(const std::string& path){
    try {
        dnnRefineModel=cv::dnn::readNetFromONNX(path);
        dnnRefineLoaded=true;
        std::cout << "Refinement ONNX loaded: " << path << "\n";
    } catch(const cv::Exception& e){
        std::cerr << "Failed to load refinement ONNX: " << e.what() << "\n";
        dnnRefineLoaded=false;
    }
}

// ============================================================================
// 核心处理流水线
// ============================================================================
cv::Mat MotionDetector::nextFrame()
{
    cv::Mat frame;
    if(!cap.read(frame)||frame.empty()) return cv::Mat();

    // 输入缩放：maxDim>0 且分辨率超出时缩小处理
    if(p.maxDim>0 && (frame.rows>p.maxDim || frame.cols>p.maxDim)){
        double s=(double)p.maxDim/std::max(frame.rows,frame.cols);
        cv::resize(frame,frame,cv::Size(),s,s,cv::INTER_AREA);
    }
    preprocess(frame);
    rawFrame=frame.clone();

    if(!initialized){
        initFrames.push_back(grayFrame.clone());
        if((int)initFrames.size()>=p.vibeN){
            initViBe(initFrames);
            initViBeLBSP(initFrames);
            initialized=true;
            previousGray=grayFrame.clone();
            initFrames.clear();
        }
        frameCount++; currentMask=cv::Mat::zeros(rows,cols,CV_8UC1);
        overlay=frame.clone();
        // 初始化阶段也需要还原输出分辨率
        if(p.maxDim>0 && p.resizeOutput && (origRows!=rows || origCols!=cols)){
            cv::resize(currentMask,currentMask,cv::Size(origCols,origRows),0,0,cv::INTER_NEAREST);
            cv::resize(overlay,overlay,cv::Size(origCols,origRows),0,0,cv::INTER_LINEAR);
            cv::resize(rawFrame,rawFrame,cv::Size(origCols,origRows),0,0,cv::INTER_LINEAR);
        }
        return currentMask;
    }

    // ====== 计算 LBSP 纹理描述子 ======
    lbspFrame = computeLBSP(grayFrame);

    // ====== ViBe 时间流 (RGB 3通道) ======
    cv::Mat grayConf  = computeViBeConfidence();
    cv::Mat lbspConf  = computeViBeConfidenceLBSP(lbspFrame);
    // 双通道加权融合：灰度主导纹理修正
    cv::Mat vibeConf;
    cv::addWeighted(grayConf, 1.0-p.lbspWeight, lbspConf, p.lbspWeight, 0, vibeConf);

    // ====== 运动感知空间流 (光流 + 帧差融合) ======
    cv::Mat motionNorm=computeMotionNorm();
    cv::Mat stdNorm=computeLocalStdNorm();
    cachedStdNorm=stdNorm;  // Cache for ONNX head / intermediate dump
    cv::Mat spatialConf=computeSpatialConfidence(motionNorm,stdNorm);

    // ====== 融合 ======
    fusionMap=fuseStreams(vibeConf,spatialConf);
    cv::Mat rawMask;

    // ====== 中间特征导出 (训练 binarization head) ======
    if(p.dumpIntermediates && !p.dumpDir.empty() && frameCount > p.vibeN){
        int idx = frameCount - p.vibeN - 1;
        char buf[256];
        snprintf(buf,sizeof(buf),"%s/fusion_%05d.tiff",p.dumpDir.c_str(),idx);
        cv::imwrite(buf, fusionMap);
        snprintf(buf,sizeof(buf),"%s/temporal_%05d.tiff",p.dumpDir.c_str(),idx);
        cv::imwrite(buf, vibeConf);
        snprintf(buf,sizeof(buf),"%s/std_%05d.tiff",p.dumpDir.c_str(),idx);
        cv::imwrite(buf, cachedStdNorm);
    }

    // ====== 二值分类: ONNX 头 OR 硬阈值 ======
    if(dnnLoaded){
        // Build [1,3,H,W] blob at native resolution
        std::vector<cv::Mat> ch3 = {fusionMap, vibeConf, cachedStdNorm};
        cv::Mat blob3ch; cv::merge(ch3, blob3ch);       // [H,W,3]
        cv::Mat blob4d = cv::dnn::blobFromImages(
            std::vector<cv::Mat>{blob3ch}, 1.0, cv::Size(), cv::Scalar(),
            false, false);                               // [1,3,H,W]

        dnnModel.setInput(blob4d);
        cv::Mat output = dnnModel.forward();             // [1,1,H,W]

        cv::Mat probMap(output.size[2], output.size[3], CV_32FC1,
                        output.ptr<float>());
        cv::threshold(probMap, rawMask, 0.5, 255, cv::THRESH_BINARY);
        rawMask.convertTo(rawMask, CV_8UC1);
    } else {
        double theta = p.lambda * 1.0 / p.vibeN;
        rawMask = binaryClassify(fusionMap, vibeConf, theta);
    }

    // ====== V9-A: 光流幅值门控 (过滤微颤/树叶等微小运动) ======
    if(p.flowGateThresh > 0 && !motionNorm.empty()){
        cv::Mat flowMask(rawMask.size(), CV_8UC1, cv::Scalar(0));
        for(int r = 0; r < rows; r++){
            const float* mRow = motionNorm.ptr<float>(r);
            uchar* fRow = flowMask.ptr<uchar>(r);
            for(int c = 0; c < cols; c++)
                if(mRow[c] >= p.flowGateThresh) fRow[c] = 255;
        }
        cv::bitwise_and(rawMask, flowMask, rawMask);
    }

    // ====== Dense CRF (替代ICM, 利用颜色信息精确贴合边缘) ======
    cv::Mat optMask=denseCRF(rawMask,fusionMap,rawFrame);

    // ====== 阴影过滤 ======
    cv::Mat shadowed;
    if(p.shadowFilter) shadowed=shadowFilter(optMask); else shadowed=optMask;

    // ====== 颜色精修 (Phase 2): 抑制颜色接近BG的疑似FP ======
    if(p.colorRefine) shadowed = colorRefine(shadowed);

    // ====== 静态像素过滤: 帧差极小→强制BG (拦截静止纹理误检) ======
    if(p.staticThresh>0) shadowed = staticFilter(shadowed);

    // ====== 光流一致性: 随机方向→树叶/噪声→抑制 ======
    if(p.flowConsistency) shadowed = flowConsistencyFilter(shadowed);

    // ====== 精修网络推理 (替代 flowWarpAlign + ghostSuppress + fillHoles + edgeSnap) ======
    if(dnnRefineLoaded && frameCount > p.vibeN && !previousMask.empty()){
        // Downscale >720p, pad to multiple of 4 for UNet skip connections
        const int MAX_DIM = 720;
        int rH = rows, rW = cols;
        double scale = 1.0;
        if(rows > MAX_DIM || cols > MAX_DIM){
            scale = (double)MAX_DIM / std::max(rows, cols);
            rH = (int)(rows * scale);
            rW = (int)(cols * scale);
        }
        // Pad to multiple of 4
        int padH = (4 - rH % 4) % 4;
        int padW = (4 - rW % 4) % 4;
        int pH = rH + padH, pW = rW + padW;

        auto resizeTo = [&](const cv::Mat& src, int h, int w){
            if(h == rows && w == cols) return src.clone();
            cv::Mat dst; cv::resize(src, dst, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
            return dst;
        };

        // Prepare 5 channels: resize to (rH,rW) then pad to (pH,pW)
        auto resizePad = [&](const cv::Mat& src, int h, int w, int ph, int pw){
            cv::Mat tmp = resizeTo(src, h, w);
            if(h == ph && w == pw) return tmp;
            cv::Mat padded; cv::copyMakeBorder(tmp, padded, 0, ph-h, 0, pw-w,
                                              cv::BORDER_REPLICATE);
            return padded;
        };

        cv::Mat maskFloat; shadowed.convertTo(maskFloat, CV_32F, 1.0/255.0);
        maskFloat = resizePad(maskFloat, rH, rW, pH, pW);

        cv::Mat grayNorm; grayFrame.convertTo(grayNorm, CV_32F, 1.0/255.0);
        grayNorm = resizePad(grayNorm, rH, rW, pH, pW);

        cv::Mat flowMag(pH, pW, CV_32FC1, cv::Scalar(0));
        if(!cachedFlowX.empty() && !cachedFlowY.empty()){
            cv::Mat fx, fy;
            cv::resize(cachedFlowX, fx, cv::Size(rW, rH), 0, 0, cv::INTER_LINEAR);
            cv::resize(cachedFlowY, fy, cv::Size(rW, rH), 0, 0, cv::INTER_LINEAR);
            fx.convertTo(fx, CV_32F); fy.convertTo(fy, CV_32F);
            cv::Mat mag2; cv::magnitude(fx, fy, mag2);
            cv::Scalar meanMag, stdMag;
            cv::meanStdDev(mag2, meanMag, stdMag);
            double maxVal = meanMag[0] + 3.0 * stdMag[0];
            if(maxVal > 1e-6) mag2 /= maxVal;
            cv::min(mag2, 1.0, flowMag);
            // Pad flowMag
            if(padH || padW){
                cv::Mat tmp = flowMag;
                cv::copyMakeBorder(tmp, flowMag, 0, padH, 0, padW, cv::BORDER_REPLICATE);
            }
        }

        cv::Mat frameDiff;
        if(!previousGray.empty()){
            cv::Mat diff; cv::absdiff(grayFrame, previousGray, diff);
            cv::Mat diffF; diff.convertTo(diffF, CV_32F, 1.0/255.0);
            frameDiff = resizePad(diffF, rH, rW, pH, pW);
        } else frameDiff = cv::Mat::zeros(pH, pW, CV_32FC1);

        cv::Mat prevMaskNorm;
        previousMask.convertTo(prevMaskNorm, CV_32F, 1.0/255.0);
        prevMaskNorm = resizePad(prevMaskNorm, rH, rW, pH, pW);

        // Manual NCHW blob: [1,5,pH,pW]
        int sz[] = {1, 5, pH, pW};
        cv::Mat blob4d(4, sz, CV_32FC1);
        float* dst = blob4d.ptr<float>();
        std::vector<cv::Mat> ch5 = {maskFloat, grayNorm, flowMag, frameDiff, prevMaskNorm};
        for(int c = 0; c < 5; c++){
            const float* src = ch5[c].ptr<float>();
            memcpy(dst + c * pH * pW, src, pH * pW * sizeof(float));
        }

        dnnRefineModel.setInput(blob4d);
        cv::Mat output = dnnRefineModel.forward();
        cv::Mat probMap(output.size[2], output.size[3], CV_32FC1, output.ptr<float>());

        // Crop padding off, then resize back to original
        if(padH || padW)
            probMap = probMap(cv::Rect(0, 0, rW, rH)).clone();
        if(scale != 1.0)
            cv::resize(probMap, probMap, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);

        cv::threshold(probMap, shadowed, 0.5, 255, cv::THRESH_BINARY);
        shadowed.convertTo(shadowed, CV_8UC1);
    }

    // ====== 精修网络训练数据导出 (原始mask + 4辅助通道) ======
    if(p.dumpRefinement && !p.refinementDir.empty() && frameCount > p.vibeN){
        int idx = frameCount - p.vibeN - 1;
        char buf[256];

        // 通道1: 原始mask (0/255 → 0.0/1.0 浮点)
        cv::Mat maskFloat; shadowed.convertTo(maskFloat, CV_32F, 1.0/255.0);

        // 通道2: 灰度图归一化
        cv::Mat grayNorm; grayFrame.convertTo(grayNorm, CV_32F, 1.0/255.0);

        // 通道3: 光流幅值
        cv::Mat flowMag(rows, cols, CV_32FC1, cv::Scalar(0));
        if(!cachedFlowX.empty() && !cachedFlowY.empty()){
            cv::Mat fx, fy;
            cachedFlowX.convertTo(fx, CV_32F);
            cachedFlowY.convertTo(fy, CV_32F);
            cv::Mat mag2; cv::magnitude(fx, fy, mag2);
            cv::Scalar meanMag, stdMag;
            cv::meanStdDev(mag2, meanMag, stdMag);
            double maxVal = meanMag[0] + 3.0 * stdMag[0];
            if(maxVal > 1e-6) mag2 /= maxVal;
            cv::min(mag2, 1.0, flowMag);
        }

        // 通道4: 帧差归一化
        cv::Mat frameDiff;
        if(!previousGray.empty()){
            cv::Mat diff; cv::absdiff(grayFrame, previousGray, diff);
            diff.convertTo(frameDiff, CV_32F, 1.0/255.0);
        } else {
            frameDiff = cv::Mat::zeros(rows, cols, CV_32FC1);
        }

        // 通道5: 上一帧mask (0/255 → 0.0/1.0)
        cv::Mat prevMaskNorm;
        if(!previousMask.empty())
            previousMask.convertTo(prevMaskNorm, CV_32F, 1.0/255.0);
        else
            prevMaskNorm = cv::Mat::zeros(rows, cols, CV_32FC1);

        // Save as raw float32 binary (avoids TIFF corruption issues)
        auto saveFloatMat = [&](const cv::Mat& m, const char* suffix){
            snprintf(buf,sizeof(buf),"%s/%s_%05d.bin",p.refinementDir.c_str(),suffix,idx);
            FILE* fp = fopen(buf, "wb");
            if(fp){
                int h = m.rows, w = m.cols;
                fwrite(&h, sizeof(int), 1, fp);
                fwrite(&w, sizeof(int), 1, fp);
                fwrite(m.ptr<float>(), sizeof(float), h*w, fp);
                fclose(fp);
            }
        };
        saveFloatMat(maskFloat, "rawmask");
        saveFloatMat(grayNorm, "gray");
        saveFloatMat(flowMag, "flowmag");
        saveFloatMat(frameDiff, "framediff");
        saveFloatMat(prevMaskNorm, "prevmask");
    }

    // ====== 光流前推对齐 (消除时序滞后) ======
    if(p.flowWarpAlign && !previousMask.empty())
        shadowed = flowWarpAlign(shadowed);

    // ====== 鬼影抑制 (静止前景+低纹理加速遗忘) ======
    if(p.ghostSuppress && !previousMask.empty())
        shadowed = ghostSuppressFilter(shadowed);

    // ====== V9-D: 帧差互补检测 (颜色相近物体 ViBe 漏检补偿) ======
    if(p.frameDiffFill && !previousGray.empty()){
        cv::Mat frameDiff;
        cv::absdiff(grayFrame, previousGray, frameDiff);
        cv::Mat diffBin;
        cv::threshold(frameDiff, diffBin, p.frameDiffThresh, 255, cv::THRESH_BINARY);
        // 对帧差检测到的像素：如果周围 FG 密度足够 → 很可能 ViBe 漏检
        cv::Mat diffComplement = cv::Mat::zeros(rows, cols, CV_8UC1);
        for(int r = 2; r < rows-2; r++){
            const uchar* dRow = diffBin.ptr<uchar>(r);
            const uchar* sRow = shadowed.ptr<uchar>(r);
            uchar* cRow = diffComplement.ptr<uchar>(r);
            for(int c = 2; c < cols-2; c++){
                if(dRow[c] == 0 || sRow[c] == 255) continue; // 帧差无变化 或 已被检测
                // 统计 5x5 邻域 FG 密度
                int fgCnt = 0;
                for(int dr = -2; dr <= 2; dr++)
                    for(int dc = -2; dc <= 2; dc++)
                        if(shadowed.ptr<uchar>(r+dr)[c+dc] == 255) fgCnt++;
                if(fgCnt >= 25 * p.frameDiffFgDensity)
                    cRow[c] = 255; // 周围已有 FG → 补充进来
            }
        }
        cv::bitwise_or(shadowed, diffComplement, shadowed);
    }

    // ====== V9-G: 距离变换空洞填充 (拓扑→几何, 修复非闭合内部孔洞) ======
    if(p.distTransformFill){
        cv::Mat dist;
        cv::distanceTransform(shadowed, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        // dist=0 在 BG 上, dist>0 在 FG 上, dist 值 = 到最近 BG 的距离
        // 所有 dist > distFillRadius 的像素 → "核心前景", 向外膨胀 fillRadius
        cv::Mat coreFg;
        cv::threshold(dist, coreFg, p.distFillRadius, 255, cv::THRESH_BINARY);
        coreFg.convertTo(coreFg, CV_8UC1);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
            cv::Size((int)(p.distFillRadius*2+1), (int)(p.distFillRadius*2+1)));
        cv::Mat dilated;
        cv::dilate(coreFg, dilated, kernel);
        cv::bitwise_or(shadowed, dilated, shadowed);
    }

    // ====== 轮廓填充 + 边缘对齐 + 二次填充 ======
    cv::Mat filled = fillHoles(shadowed);
    filled = edgeSnap(filled, rawFrame);
    filled = fillHoles(filled);  // edgeSnap可能产生新孔, 重新填充

    // ====== 边界腐蚀+二次填充 (收缩边界提升Precision, 重填恢复内部) ======
    if(p.erodeSize>0){
        cv::Mat erodeKernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,
            cv::Size(p.erodeSize*2+1,p.erodeSize*2+1));
        cv::erode(filled,filled,erodeKernel);
        // 二次填充: 腐蚀只去除外部边界, 内部被误删的像素在轮廓填充中恢复
        filled = fillHoles(filled);
    }

    // ====== 时序滤波 ======
    cv::Mat tempFiltered;
    if(p.temporalMedian) tempFiltered=temporalMedianFilter(filled);
    else tempFiltered=filled;

    // ====== 连通域过滤 ======
    currentMask=filterSmallComponents(tempFiltered, fusionMap);

    // ====== 人体碎片修复 V8: 自适应morph + 运动连通 + 主体膨胀吸附 + 光流过滤 ======
    if(!currentMask.empty() && cv::countNonZero(currentMask) > 0){
        cv::Mat labels, stats, cents;
        int nLabels = cv::connectedComponentsWithStats(currentMask, labels, stats, cents, 8, CV_32S);
        if(nLabels > 1){
            cv::Mat repairedMask = currentMask.clone();
            bool hasFlow = !cachedFlowX.empty() && !cachedFlowY.empty();

            // Step 1: 自适应形态学核 (细长→大CLOSE, 块状→小CLOSE)
            for(int i = 1; i < nLabels; i++){
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if(area < p.minArea) continue;
                int x = stats.at<int>(i, cv::CC_STAT_LEFT);
                int y = stats.at<int>(i, cv::CC_STAT_TOP);
                int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
                int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                x = std::max(0, x-5); y = std::max(0, y-5);
                w = std::min(cols-x, w+10); h = std::min(rows-y, h+10);
                cv::Rect roi(x, y, w, h);
                cv::Mat blobMask = (labels(roi) == i);
                cv::Mat blobU8; blobMask.convertTo(blobU8, CV_8UC1, 255);

                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(blobU8, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                double circularity = 0.03;
                if(!contours.empty()){
                    double peri = cv::arcLength(contours[0], true);
                    if(peri > 1.0)
                        circularity = 4.0 * CV_PI * area / (peri * peri);
                }
                int ksize = (circularity < 0.06) ? 7 : 3;
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
                cv::Mat localPatch = repairedMask(roi);
                cv::Mat closed;
                cv::morphologyEx(localPatch, closed, cv::MORPH_CLOSE, kernel);
                closed.copyTo(localPatch);
            }

            // Step 2: 运动矢量连通 (光流相似的近邻碎片画桥)
            if(hasFlow){
                struct BlobInfo { cv::Point2f centroid, flow; int area, id; };
                std::vector<BlobInfo> blobs;
                for(int i = 1; i < nLabels; i++){
                    int area = stats.at<int>(i, cv::CC_STAT_AREA);
                    if(area < p.minArea) continue;
                    cv::Point2f cent(cents.at<double>(i, 0), cents.at<double>(i, 1));
                    float fx=0, fy=0; int cnt=0;
                    for(int r = 0; r < rows; r++){
                        const int* lbl = labels.ptr<int>(r);
                        const float* fx_r = cachedFlowX.ptr<float>(r);
                        const float* fy_r = cachedFlowY.ptr<float>(r);
                        for(int c = 0; c < cols; c++){
                            if(lbl[c] == i){ fx+=fx_r[c]; fy+=fy_r[c]; cnt++; }
                        }
                    }
                    if(cnt>0){ fx/=cnt; fy/=cnt; }
                    blobs.push_back({cent, {fx,fy}, area, i});
                }

                cv::Mat bridgeMask = cv::Mat::zeros(rows, cols, CV_8UC1);
                for(size_t a = 0; a < blobs.size(); a++){
                    for(size_t b = a+1; b < blobs.size(); b++){
                        float dx = blobs[a].centroid.x - blobs[b].centroid.x;
                        float dy = blobs[a].centroid.y - blobs[b].centroid.y;
                        float dist = std::sqrt(dx*dx+dy*dy);
                        if(dist < 1 || dist > 30) continue;
                        float magA = std::sqrt(blobs[a].flow.x*blobs[a].flow.x +
                                               blobs[a].flow.y*blobs[a].flow.y);
                        float magB = std::sqrt(blobs[b].flow.x*blobs[b].flow.x +
                                               blobs[b].flow.y*blobs[b].flow.y);
                        if(magA < 0.1 || magB < 0.1) continue;
                        float dot = blobs[a].flow.x*blobs[b].flow.x +
                                    blobs[a].flow.y*blobs[b].flow.y;
                        float cosAngle = dot / (magA*magB + 1e-6f);
                        float magRatio = magA / (magB + 1e-6f);
                        if(cosAngle > 0.9 && magRatio > 0.3 && magRatio < 3.0){
                            cv::line(bridgeMask, cv::Point(blobs[a].centroid),
                                     cv::Point(blobs[b].centroid), cv::Scalar(255), 5);
                        }
                    }
                }
                cv::bitwise_or(repairedMask, bridgeMask, repairedMask);

                int bigThresh = p.minArea * 3; // 主体大小阈值

                // Step 3: 主体膨胀吸附 (大blob膨胀找近邻碎片→直接合并碎片, 不动主体边界)
                cv::Mat labels2, stats2, cents2;
                int n2 = cv::connectedComponentsWithStats(repairedMask, labels2, stats2, cents2, 8, CV_32S);
                if(n2 > 1){
                    // 收集大blob的膨胀区域
                    cv::Mat allDilated = cv::Mat::zeros(rows, cols, CV_8UC1);
                    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
                    for(int i = 1; i < n2; i++){
                        if(stats2.at<int>(i, cv::CC_STAT_AREA) < bigThresh) continue;
                        cv::Mat blobU; cv::Mat(labels2 == i).convertTo(blobU, CV_8UC1, 255);
                        cv::Mat dilated;
                        cv::dilate(blobU, dilated, dilateKernel);
                        cv::bitwise_or(allDilated, dilated, allDilated);
                    }
                    // 碎片落在膨胀区内 → 合并碎片(不合并膨胀区,避免扩边)
                    for(int i = 1; i < n2; i++){
                        int area = stats2.at<int>(i, cv::CC_STAT_AREA);
                        if(area >= bigThresh || area < p.minArea) continue;
                        cv::Mat blobM = (labels2 == i);
                        cv::Mat overlap; cv::bitwise_and(blobM, allDilated, overlap);
                        if(cv::countNonZero(overlap) > area * 0.2){
                            cv::Mat blobU; blobM.convertTo(blobU, CV_8UC1, 255);
                            cv::bitwise_or(repairedMask, blobU, repairedMask);
                        }
                    }
                }

                // Step 4: 光流方向过滤 (残块有邻近大blob→比较方向; 无邻近大blob→保留为独立物体)
                cv::Mat labels3, stats3, cents3;
                int n3 = cv::connectedComponentsWithStats(repairedMask, labels3, stats3, cents3, 8, CV_32S);
                if(n3 > 1){
                    // 收集大blob信息
                    std::vector<std::pair<cv::Point2f, cv::Point2f>> bigBlobs;
                    for(int i = 1; i < n3; i++){
                        int area = stats3.at<int>(i, cv::CC_STAT_AREA);
                        if(area < bigThresh) continue;
                        cv::Point2f cent(cents3.at<double>(i, 0), cents3.at<double>(i, 1));
                        float fx=0, fy=0; int cnt=0;
                        for(int r = 0; r < rows; r++){
                            const int* lbl = labels3.ptr<int>(r);
                            const float* fx_r = cachedFlowX.ptr<float>(r);
                            const float* fy_r = cachedFlowY.ptr<float>(r);
                            for(int c = 0; c < cols; c++)
                                if(lbl[c] == i){ fx+=fx_r[c]; fy+=fy_r[c]; cnt++; }
                        }
                        if(cnt>0){ fx/=cnt; fy/=cnt; }
                        bigBlobs.push_back({cent, {fx, fy}});
                    }

                    for(int i = 1; i < n3; i++){
                        int area = stats3.at<int>(i, cv::CC_STAT_AREA);
                        if(area >= bigThresh || area < p.minArea) continue;
                        cv::Point2f cent(cents3.at<double>(i, 0), cents3.at<double>(i, 1));
                        float fx=0, fy=0; int cnt=0;
                        for(int r = 0; r < rows; r++){
                            const int* lbl = labels3.ptr<int>(r);
                            const float* fx_r = cachedFlowX.ptr<float>(r);
                            const float* fy_r = cachedFlowY.ptr<float>(r);
                            for(int c = 0; c < cols; c++)
                                if(lbl[c] == i){ fx+=fx_r[c]; fy+=fy_r[c]; cnt++; }
                        }
                        if(cnt == 0) continue; // 无光流信息→保留
                        fx/=cnt; fy/=cnt;
                        float magF = std::sqrt(fx*fx+fy*fy);
                        if(magF < 0.1) continue; // 静止碎片→保留(可能是停车/小物体)

                        // 找最近的大blob
                        float bestDist = 1e9; int bestIdx = -1;
                        for(size_t b = 0; b < bigBlobs.size(); b++){
                            float dx = cent.x - bigBlobs[b].first.x;
                            float dy = cent.y - bigBlobs[b].first.y;
                            float d = std::sqrt(dx*dx+dy*dy);
                            if(d < bestDist){ bestDist = d; bestIdx = (int)b; }
                        }

                        // 附近没有大blob → 保留为独立物体
                        if(bestIdx < 0 || bestDist > 50) continue;

                        float magB = std::sqrt(bigBlobs[bestIdx].second.x*bigBlobs[bestIdx].second.x +
                                               bigBlobs[bestIdx].second.y*bigBlobs[bestIdx].second.y);
                        if(magB < 0.1) continue; // 大blob静止 → 不判断

                        float dot = fx*bigBlobs[bestIdx].second.x + fy*bigBlobs[bestIdx].second.y;
                        float cosAng = dot / (magF*magB + 1e-6f);

                        if(cosAng < 0.5){
                            // 方向明显不同且靠近大blob → 噪声, 删除
                            repairedMask.setTo(0, (labels3 == i));
                        }
                        // 方向一致 → 保留 (不做额外合并, Step3已处理)
                    }
                }
            } // hasFlow

            currentMask = repairedMask;
        }
    }

    // ====== V9-B: 时序持续性过滤 (像素必须持续FG多帧才确认为真正前景) ======
    if(p.temporalPersistence){
        if(persistenceCount.empty())
            persistenceCount = cv::Mat::zeros(rows, cols, CV_32SC1);
        for(int r = 0; r < rows; r++){
            const uchar* mRow = currentMask.ptr<uchar>(r);
            int* pRow = persistenceCount.ptr<int>(r);
            for(int c = 0; c < cols; c++){
                if(mRow[c] == 255)
                    pRow[c] = std::min(pRow[c] + 1, p.persistenceFrames + 5);
                else
                    pRow[c] = std::max(0, pRow[c] - p.persistenceDecay);
            }
        }
        // 计数器达标的像素才会保留
        for(int r = 0; r < rows; r++){
            uchar* mRow = currentMask.ptr<uchar>(r);
            const int* pRow = persistenceCount.ptr<int>(r);
            for(int c = 0; c < cols; c++)
                if(pRow[c] < p.persistenceFrames) mRow[c] = 0;
        }
    } else {
        // 不用持久化则清零计数器
        if(!persistenceCount.empty()) persistenceCount = cv::Mat();
    }

    // ViBe更新(用后处理前的mask避免污染)
    updateViBe(shadowed);
    updateViBeLBSP(lbspFrame,shadowed);

    buildOverlay(currentMask);
    previousMask=currentMask.clone();
    previousGray=grayFrame.clone();
    previousColor=rawFrame.clone();

    // 输出还原：如果做了输入缩放且 resizeOutput 为 true
    if(p.maxDim>0 && p.resizeOutput && (origRows!=rows || origCols!=cols)){
        cv::resize(currentMask, currentMask, cv::Size(origCols, origRows), 0, 0, cv::INTER_NEAREST);
        cv::resize(overlay, overlay, cv::Size(origCols, origRows), 0, 0, cv::INTER_LINEAR);
        cv::resize(rawFrame, rawFrame, cv::Size(origCols, origRows), 0, 0, cv::INTER_LINEAR);
    }
    frameCount++; return currentMask;
}

// ============================================================================
void MotionDetector::preprocess(const cv::Mat& frame){
    // 3通道模糊: RGB各通道独立去噪，保留颜色信息供ViBe使用
    if(frame.channels()==3){
        cv::GaussianBlur(frame,colorFrame,
            cv::Size(p.gaussianKernel,p.gaussianKernel),p.gaussianSigma);
        cv::cvtColor(colorFrame,grayFrame,cv::COLOR_BGR2GRAY);
    } else {
        cv::GaussianBlur(frame,grayFrame,
            cv::Size(p.gaussianKernel,p.gaussianKernel),p.gaussianSigma);
        cv::cvtColor(grayFrame,colorFrame,cv::COLOR_GRAY2BGR);
    }
}

// ============================================================================
// ViBe — 灰度通道
// ============================================================================
void MotionDetector::initViBe(const std::vector<cv::Mat>& frames){
    const int N=(int)frames.size();
    if(p.useRGB){
        // RGB 模式: 预模糊所有初始帧 → bgSamples=CV_8UC(N*3) BGR交错
        std::vector<cv::Mat> blurred(N);
        for(int n=0;n<N;n++){
            if(frames[n].channels()==3){
                cv::GaussianBlur(frames[n],blurred[n],
                    cv::Size(p.gaussianKernel,p.gaussianKernel),p.gaussianSigma);
            } else {
                cv::GaussianBlur(frames[n],blurred[n],
                    cv::Size(p.gaussianKernel,p.gaussianKernel),p.gaussianSigma);
                cv::cvtColor(blurred[n],blurred[n],cv::COLOR_GRAY2BGR);
            }
        }
        bgSamples=cv::Mat(rows,cols,CV_8UC(N*3));
        for(int r=0;r<rows;r++){ uchar* sr=bgSamples.ptr<uchar>(r);
            for(int c=0;c<cols;c++){
                for(int n=0;n<N;n++){
                    cv::Vec3b pix=blurred[rng()%N].at<cv::Vec3b>(r,c);
                    int base=(c*N+n)*3;
                    sr[base+0]=pix[0]; sr[base+1]=pix[1]; sr[base+2]=pix[2];
                }
            }
        }
    } else {
        // 灰度模式
        bgSamples=cv::Mat(rows,cols,CV_8UC(N));
        for(int r=0;r<rows;r++){ uchar* sr=bgSamples.ptr<uchar>(r);
            for(int c=0;c<cols;c++)
                for(int n=0;n<N;n++) sr[c*N+n]=frames[rng()%N].at<uchar>(r,c);
        }
    }
    // bgDisplay: 3通道 (RGB模式取中位数, 灰度模式扩展)
    bgDisplay=cv::Mat(rows,cols,CV_8UC3);
    if(p.useRGB){
        for(int r=0;r<rows;r++){ const uchar* sr=bgSamples.ptr<uchar>(r); cv::Vec3b* dr=bgDisplay.ptr<cv::Vec3b>(r);
            for(int c=0;c<cols;c++){ std::vector<uchar> b(N),g(N),r_(N);
                for(int n=0;n<N;n++){ int base=(c*N+n)*3;
                    b[n]=sr[base+0]; g[n]=sr[base+1]; r_[n]=sr[base+2]; }
                std::sort(b.begin(),b.end()); std::sort(g.begin(),g.end()); std::sort(r_.begin(),r_.end());
                dr[c]=cv::Vec3b(b[N/2],g[N/2],r_[N/2]);
            }
        }
    } else {
        for(int r=0;r<rows;r++){ const uchar* sr=bgSamples.ptr<uchar>(r); cv::Vec3b* dr=bgDisplay.ptr<cv::Vec3b>(r);
            for(int c=0;c<cols;c++){ std::vector<uchar> v(N);
                for(int n=0;n<N;n++) v[n]=sr[c*N+n];
                std::sort(v.begin(),v.end());
                dr[c]=cv::Vec3b(v[N/2],v[N/2],v[N/2]);
            }
        }
    }
    // perPixelStd 从灰度估计
    ghostTimer=cv::Mat::zeros(rows,cols,CV_32SC1);
    perPixelStd=cv::Mat(rows,cols,CV_32FC1);
    for(int r=0;r<rows;r++){ float* ps=perPixelStd.ptr<float>(r);
        const uchar* gr=grayFrame.ptr<uchar>(r);
        for(int c=0;c<cols;c++){
            float sum=0,sq=0; int Ns=std::min(N,10);
            for(int n=0;n<Ns;n++){
                float v=p.useRGB?(float)bgSamples.ptr<uchar>(r)[(c*N+n)*3+1]: // G channel approx
                                (float)bgSamples.ptr<uchar>(r)[c*N+n];
                sum+=v; sq+=v*v;
            }
            float mean=sum/Ns,var=sq/Ns-mean*mean;
            ps[c]=std::sqrt(std::max(var,0.0f));
        }
    }
}

cv::Mat MotionDetector::computeViBeConfidence(){
    const int N=p.vibeN,baseR=p.vibeR;
    const double k=p.adaptRK;
    cv::Mat conf(rows,cols,CV_32FC1);
    for(int r=0;r<rows;r++){ const uchar* sr=bgSamples.ptr<uchar>(r);
        const float* ps=perPixelStd.ptr<float>(r); float* cf=conf.ptr<float>(r);
        for(int c=0;c<cols;c++){
            int adaptR=baseR;
            if(p.adaptiveR){
                adaptR=cvRound(baseR+k*ps[c]);
                adaptR=std::max(p.adaptRMin,std::min(p.adaptRMax,adaptR));
            }
            int m=0;
            if(p.useRGB){
                cv::Vec3b cur=colorFrame.ptr<cv::Vec3b>(r)[c];
                int baseIdx=(c*N)*3;
                int chR=cvRound(adaptR*p.rgbRScale);
                for(int n=0;n<N;n++){
                    int idx=baseIdx+n*3;
                    if(std::abs((int)sr[idx+0]-(int)cur[0])<chR &&
                       std::abs((int)sr[idx+1]-(int)cur[1])<chR &&
                       std::abs((int)sr[idx+2]-(int)cur[2])<chR) m++;
                }
            } else {
                uchar cur=grayFrame.ptr<uchar>(r)[c];
                for(int n=0;n<N;n++)
                    if(std::abs((int)sr[c*N+n]-(int)cur)<adaptR) m++;
            }
            cf[c]=1.0f-(float)m/N;
        }
    }
    return conf;
}

void MotionDetector::updateViBe(const cv::Mat& mask){
    const int N=p.vibeN,R=p.vibeR,minM=p.vibeMin,phi=p.vibePhi;
    for(int r=0;r<rows;r++){ uchar* sr=bgSamples.ptr<uchar>(r);
        for(int c=0;c<cols;c++){
            int m=0;
            if(p.useRGB){
                cv::Vec3b cur=colorFrame.ptr<cv::Vec3b>(r)[c];
                int baseIdx=(c*N)*3;
                int chR=cvRound(R*p.rgbRScale);
                for(int n=0;n<N;n++){
                    int idx=baseIdx+n*3;
                    if(std::abs((int)sr[idx+0]-(int)cur[0])<chR &&
                       std::abs((int)sr[idx+1]-(int)cur[1])<chR &&
                       std::abs((int)sr[idx+2]-(int)cur[2])<chR) m++;
                }
                if(m>=minM){
                    if((rng()%phi)==0){
                        int idx=baseIdx+(rng()%N)*3;
                        sr[idx+0]=cur[0]; sr[idx+1]=cur[1]; sr[idx+2]=cur[2];
                    }
                    if((rng()%phi)==0){
                        int nr=r+(int)(rng()%3)-1,nc=c+(int)(rng()%3)-1;
                        if(nr>=0&&nr<rows&&nc>=0&&nc<cols){
                            int nIdx=(nc*N+(rng()%N))*3; uchar* nsr=bgSamples.ptr<uchar>(nr);
                            nsr[nIdx+0]=cur[0]; nsr[nIdx+1]=cur[1]; nsr[nIdx+2]=cur[2];
                        }
                    }
                }
            } else {
                uchar cur=grayFrame.ptr<uchar>(r)[c];
                for(int n=0;n<N;n++) if(std::abs((int)sr[c*N+n]-(int)cur)<R) m++;
                int effMinM=minM;
                if(p.ghostSuppress){
                    float localStd=perPixelStd.ptr<float>(r)[c];
                    if(localStd<(float)p.ghostLowTextureR) effMinM=0;
                }
                if(m>=effMinM){
                    if((rng()%phi)==0) sr[c*N+(rng()%N)]=cur;
                    if((rng()%phi)==0){
                        int nr=r+(int)(rng()%3)-1,nc=c+(int)(rng()%3)-1;
                        if(nr>=0&&nr<rows&&nc>=0&&nc<cols) bgSamples.ptr<uchar>(nr)[nc*N+(rng()%N)]=cur;
                    }
                }
            }
        }
    }
}

// ============================================================================
// ViBe — LBSP 纹理通道 (Phase 1)
// ============================================================================
cv::Mat MotionDetector::computeLBSP(const cv::Mat& gray){
    // 2个半径×8邻域 = 16-bit 描述子，对光照变化不变
    const int T=p.lbspT;
    // 内圈 (radius=1) 和外圈 (radius=3) 的8方向偏移
    const int dr1[8]={-1,-1, 0, 1, 1, 1, 0,-1};
    const int dc1[8]={ 0, 1, 1, 1, 0,-1,-1,-1};
    const int dr2[8]={-3,-3, 0, 3, 3, 3, 0,-3};
    const int dc2[8]={ 0, 3, 3, 3, 0,-3,-3,-3};

    cv::Mat lbsp(rows,cols,CV_16UC1);
    for(int r=0;r<rows;r++){ const uchar* gr=gray.ptr<uchar>(r); uint16_t* lr=lbsp.ptr<uint16_t>(r);
        for(int c=0;c<cols;c++){
            uchar center=gr[c]; uint16_t desc=0;
            // 内圈 8 位 (bits 0-7)
            for(int i=0;i<8;i++){
                int nr=r+dr1[i],nc=c+dc1[i];
                // 边界处理：镜像扩展
                nr=std::max(0,std::min(rows-1,nr));
                nc=std::max(0,std::min(cols-1,nc));
                if(std::abs((int)gray.at<uchar>(nr,nc)-(int)center)<T)
                    desc|=(1<<i);
            }
            // 外圈 8 位 (bits 8-15)
            for(int i=0;i<8;i++){
                int nr=r+dr2[i],nc=c+dc2[i];
                nr=std::max(0,std::min(rows-1,nr));
                nc=std::max(0,std::min(cols-1,nc));
                if(std::abs((int)gray.at<uchar>(nr,nc)-(int)center)<T)
                    desc|=(1<<(i+8));
            }
            lr[c]=desc;
        }
    }
    return lbsp;
}

void MotionDetector::initViBeLBSP(const std::vector<cv::Mat>& frames){
    const int N=(int)frames.size();
    // 为每帧预先计算 LBSP 描述子
    std::vector<cv::Mat> lbspFrames(N);
    for(int n=0;n<N;n++) lbspFrames[n]=computeLBSP(frames[n]);

    lbspBgSamples=cv::Mat(rows,cols,CV_16UC(N));
    std::mt19937 rng2(123);
    for(int r=0;r<rows;r++){ uint16_t* sr=lbspBgSamples.ptr<uint16_t>(r);
        for(int c=0;c<cols;c++){
            for(int n=0;n<N;n++){
                int fidx=rng2()%N; // 随机采样初始帧
                sr[c*N+n]=lbspFrames[fidx].at<uint16_t>(r,c);
            }
        }
    }
}

cv::Mat MotionDetector::computeViBeConfidenceLBSP(const cv::Mat& lbsp){
    const int N=p.vibeN,HammR=p.lbspHammR;
    cv::Mat conf(rows,cols,CV_32FC1);
    for(int r=0;r<rows;r++){ const uint16_t* lr=lbsp.ptr<uint16_t>(r),*sr=lbspBgSamples.ptr<uint16_t>(r); float* cr=conf.ptr<float>(r);
        for(int c=0;c<cols;c++){
            uint16_t cur=lr[c]; int m=0;
            for(int n=0;n<N;n++){
                int hdist=__builtin_popcount((unsigned int)(cur^sr[c*N+n]));
                if(hdist<=HammR) m++;
            }
            cr[c]=1.0f-(float)m/N;
        }
    }
    return conf;
}

void MotionDetector::updateViBeLBSP(const cv::Mat& lbsp, const cv::Mat& mask){
    const int N=p.vibeN,HammR=p.lbspHammR,phi=p.vibePhi,minM=p.vibeMin;
    for(int r=0;r<rows;r++){ const uint16_t* lr=lbsp.ptr<uint16_t>(r); uint16_t* sr=lbspBgSamples.ptr<uint16_t>(r);
        for(int c=0;c<cols;c++){
            uint16_t cur=lr[c]; int m=0;
            for(int n=0;n<N;n++){
                if(__builtin_popcount((unsigned int)(cur^sr[c*N+n]))<=HammR) m++;
            }
            if(m>=minM){
                if((rng()%phi)==0) sr[c*N+(rng()%N)]=cur;
                // LBSP 不扩散: 纹理描述子是位置绑定的, 扩散会污染邻居模型
            }
        }
    }
}

// ============================================================================
// 运动证据: 稠密光流 + 帧差梯度 融合
// 光流提供精确的运动边界，帧差梯度作为互补的纹理变化证据
// ============================================================================
cv::Mat MotionDetector::computeMotionNorm(){
    // ---- 1. 帧差梯度 (保留原方法，捕捉纹理变化) ----
    cv::Mat diff; cv::absdiff(grayFrame,previousGray,diff);
    cv::Mat gx,gy,gxf,gyf,gradMag;
    cv::Sobel(diff,gx,CV_16S,1,0,3);
    cv::Sobel(diff,gy,CV_16S,0,1,3);
    gx.convertTo(gxf,CV_32F); gy.convertTo(gyf,CV_32F);
    cv::magnitude(gxf,gyf,gradMag);
    double mn,mx; cv::minMaxLoc(gradMag,&mn,&mx);
    cv::Mat gNorm;
    if(mx>mn){
        cv::Mat u8; gradMag.convertTo(u8,CV_8UC1,255.0/(mx-mn),-mn*255.0/(mx-mn));
        double tg=cv::threshold(u8,u8,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
        tg=std::max(tg,1.0); cv::min(gradMag/tg,1.0,gNorm);
    } else gNorm=cv::Mat::zeros(rows,cols,CV_32FC1);

    // ---- 2. 稠密光流幅值 (Farneback, 半分辨率) ----
    cv::Mat prevSmall, currSmall;
    cv::resize(previousGray, prevSmall, cv::Size(), p.flowScale, p.flowScale, cv::INTER_LINEAR);
    cv::resize(grayFrame, currSmall, cv::Size(), p.flowScale, p.flowScale, cv::INTER_LINEAR);
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prevSmall, currSmall, flow,
        0.5, 3, 15, 3, 5, 1.2, 0);
    cv::resize(flow, flow, cv::Size(cols,rows), 0, 0, cv::INTER_LINEAR);
    flow *= 1.0/p.flowScale;
    std::vector<cv::Mat> fxy; cv::split(flow,fxy);
    cachedFlowX=fxy[0].clone(); cachedFlowY=fxy[1].clone();
    cv::Mat flowMag; cv::magnitude(fxy[0],fxy[1],flowMag);

    cv::Scalar fMean,fStd; cv::meanStdDev(flowMag,fMean,fStd);
    double fMaxNorm=fMean[0]+3.0*fStd[0];
    if(fMaxNorm>0){
        cv::Mat fNorm; cv::min(flowMag/fMaxNorm,1.0,fNorm);
        if(p.flowWeight<1.0){
            cv::Mat blend;
            cv::addWeighted(fNorm, p.flowWeight, gNorm, 1.0-p.flowWeight, 0, blend);
            return blend;
        }
        return fNorm;
    }
    return gNorm;
}


cv::Mat MotionDetector::computeLocalStdNorm(){
    cv::Mat gf; grayFrame.convertTo(gf,CV_32F); int w=p.localStdWindow;
    cv::Mat mu,msq,va,sd;
    cv::GaussianBlur(gf,mu,cv::Size(w,w),0);
    cv::Mat sq; cv::multiply(gf,gf,sq);
    cv::GaussianBlur(sq,msq,cv::Size(w,w),0);
    cv::subtract(msq,mu.mul(mu),va); cv::max(va,0,va); cv::sqrt(va,sd);
    cv::Mat n; cv::min(sd/p.tau_sigma,1.0,n); return n;
}

cv::Mat MotionDetector::computeSpatialConfidence(const cv::Mat& m,const cv::Mat& s)
{ cv::Mat r; cv::multiply(m,s,r); return r; }

// ============================================================================
cv::Mat MotionDetector::fuseStreams(const cv::Mat& t,const cv::Mat& s){
    cv::Mat c; cv::multiply(t,s,c); return p.alpha*t+p.beta*s+p.gamma*c;
}

cv::Mat MotionDetector::binaryClassify(const cv::Mat& fusion,const cv::Mat& temporal,
                                        double threshold){
    cv::Mat mask(rows,cols,CV_8UC1,cv::Scalar(0));
    for(int r=0;r<rows;r++){ const float* fr=fusion.ptr<float>(r),*tr=temporal.ptr<float>(r); uchar* mr=mask.ptr<uchar>(r);
        for(int c=0;c<cols;c++) if(fr[c]>threshold&&tr[c]>p.tau_min) mr[c]=255;
    }
    return mask;
}

// ============================================================================
// Guided Filter (Phase 1) — O(N) 边缘保持平滑，替代ICM
// ---------------------------------------------------------------------------
// 论文: He et al. "Guided Image Filtering" (ECCV 2010 / PAMI 2013)
// 原理: q = a*I + b, 其中 a = cov(I,p)/(var(I)+ε), b = mean(p)-a*mean(I)
// 实现: 全部用 cv::boxFilter 完成, O(N) 且高度 SIMD 优化
// ============================================================================
cv::Mat MotionDetector::denseCRF(const cv::Mat& binMask, const cv::Mat& fusion,
                                  const cv::Mat& colorFrame){
    // Guide: 原图灰度（边缘信息源）
    cv::Mat guide;
    if(colorFrame.channels()==3) cv::cvtColor(colorFrame,guide,cv::COLOR_BGR2GRAY);
    else guide=colorFrame.clone();
    cv::Mat I; guide.convertTo(I,CV_32F,1.0/255.0); // guide ∈ [0,1]

    // Input: 二值mask转软掩码 prob ∈ [0,1]
    cv::Mat prob;
    binMask.convertTo(prob, CV_32F, 1.0/255.0); // [0, 1]

    // Guided filter: radius, eps=regularization
    const int radius=p.gfRadius;
    const float eps=(float)p.gfEpsilon;
    const int kSize=2*radius+1;

    cv::Mat meanI, meanP, meanIP, meanII;
    cv::boxFilter(I, meanI,   CV_32F, cv::Size(kSize,kSize));
    cv::boxFilter(prob, meanP,   CV_32F, cv::Size(kSize,kSize));
    cv::boxFilter(I.mul(I), meanII, CV_32F, cv::Size(kSize,kSize));
    cv::boxFilter(I.mul(prob), meanIP, CV_32F, cv::Size(kSize,kSize));

    cv::Mat varI=meanII-meanI.mul(meanI);
    cv::Mat covIP=meanIP-meanI.mul(meanP);

    // a = covIP / (varI + eps)
    cv::Mat a,b;
    cv::divide(covIP, varI+eps, a);
    // b = meanP - a * meanI
    b=meanP-a.mul(meanI);

    // q = mean(a)*I + mean(b)  (对每个像素平均 a,b 后线性变换)
    cv::Mat meanA, meanB;
    cv::boxFilter(a, meanA, CV_32F, cv::Size(kSize,kSize));
    cv::boxFilter(b, meanB, CV_32F, cv::Size(kSize,kSize));

    cv::Mat q;
    cv::add(meanA.mul(I), meanB, q);

    // 边缘感知硬分类: 利用原图梯度定位边界，边界处加严阈值收缩mask
    cv::Mat imgGrad;
    {   // 原图 Sobel 梯度幅值 → 真实边缘位置
        cv::Mat rawGray, gx, gy;
        if(colorFrame.channels()==3) cv::cvtColor(colorFrame,rawGray,cv::COLOR_BGR2GRAY);
        else rawGray=colorFrame;
        cv::Sobel(rawGray,gx,CV_32F,1,0,3);
        cv::Sobel(rawGray,gy,CV_32F,0,1,3);
        cv::magnitude(gx,gy,imgGrad);
    }
    double gMin,gMax; cv::minMaxLoc(imgGrad,&gMin,&gMax);
    float gScale=(gMax>gMin)?(float)(1.0/(gMax-gMin)):0.0f;
    cv::Mat result(rows,cols,CV_8UC1);
    for(int r=0;r<rows;r++){ const float* qr=q.ptr<float>(r),*gr=imgGrad.ptr<float>(r); uchar* rr=result.ptr<uchar>(r);
        for(int c=0;c<cols;c++){
            float localEdge=(gr[c]-gMin)*gScale;
            float localThresh=p.gfThreshold+p.gfEdgeBias*localEdge;
            rr[c]=(qr[c]>localThresh)?255:0;
        }
    }
    return result;
}

// ============================================================================
cv::Mat MotionDetector::shadowFilter(const cv::Mat& mask){
    if(rawFrame.channels()!=3)return mask;
    cv::Mat hsv; cv::cvtColor(rawFrame,hsv,cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hc; cv::split(hsv,hc);
    cv::Mat &S=hc[1],&V=hc[2];
    // bgDisplay 是3通道BGR, 取 max(B,G,R) 近似 V(亮度)
    cv::Mat bgV(rows,cols,CV_8UC1);
    for(int r=0;r<rows;r++){ const cv::Vec3b* br=bgDisplay.ptr<cv::Vec3b>(r); uchar* bvr=bgV.ptr<uchar>(r);
        for(int c=0;c<cols;c++)
            bvr[c]=std::max({br[c][0],br[c][1],br[c][2]});
    }
    cv::Mat result=mask.clone();
    for(int r=0;r<rows;r++){ const uchar*mr=mask.ptr<uchar>(r),*sr=S.ptr<uchar>(r),*vr=V.ptr<uchar>(r),*bvr=bgV.ptr<uchar>(r); uchar*rr=result.ptr<uchar>(r);
        for(int c=0;c<cols;c++) if(mr[c]==255&&sr[c]<p.shadowSatMax&&vr[c]<bvr[c]*p.shadowValRatio)rr[c]=0;
    }
    return result;
}

// ============================================================================
cv::Mat MotionDetector::temporalMedianFilter(const cv::Mat& mask){
    maskHistory.push_back(mask.clone());
    int W=p.temporalWindow;
    while((int)maskHistory.size()>W) maskHistory.pop_front();
    if((int)maskHistory.size()<W) return mask;
    // 累加 N 帧 mask 值 → 投票
    cv::Mat sum=cv::Mat::zeros(rows,cols,CV_32FC1);
    for(int i=0;i<W;i++){
        cv::Mat tmp; maskHistory[i].convertTo(tmp,CV_32F,1.0/255.0);
        sum+=tmp;
    }
    cv::Mat r=(sum>=p.temporalVotes); r.convertTo(r,CV_8UC1,255.0);
    return r;
}

// ============================================================================
// 光流前推对齐 (消除时序滞后拖影)
// 将上一帧mask按光流向量warp到当前帧位置, 与当前mask融合
// 原理: 运动目标的mask应沿运动方向前移, 而非滞后
// ============================================================================
cv::Mat MotionDetector::flowWarpAlign(const cv::Mat& mask){
    // 用光流向量将previousMask warped到当前帧(t+1 = t + flow)
    cv::Mat warpMask(rows,cols,CV_8UC1,cv::Scalar(0));
    for(int r=0;r<rows;r++){ const uchar* pmr=previousMask.ptr<uchar>(r);
        const float* fyr=cachedFlowY.ptr<float>(r),*fxr=cachedFlowX.ptr<float>(r);
        uchar* wr=warpMask.ptr<uchar>(r);
        for(int c=0;c<cols;c++){
            if(pmr[c]==255){
                // 沿光流方向前推
                int nc=c+cvRound(fxr[c]);
                int nr=r+cvRound(fyr[c]);
                if(nr>=0&&nr<rows&&nc>=0&&nc<cols) wr[nc]=pmr[c];
            }
        }
    }
    // 侵蚀2px消除前推产生的拖尾, 再用3x3膨胀恢复主体
    cv::Mat eroded;
    cv::erode(warpMask,eroded,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));
    cv::dilate(eroded,warpMask,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5,5)));

    // 加权融合: F_out = F_curr + α*F_warped (α≤0.3, 确保当前帧主导)
    cv::Mat blend=mask.clone();
    for(int r=0;r<rows;r++){ const uchar* wr=warpMask.ptr<uchar>(r);
        uchar* br=blend.ptr<uchar>(r);
        for(int c=0;c<cols;c++){
            if(wr[c]==255 && (rng()%100)<(p.flowWarpWeight*100))
                br[c]=255;
        }
    }
    return blend;
}

// ============================================================================
// 鬼影抑制: 静止前景 + 低纹理区加速遗忘
// ============================================================================
cv::Mat MotionDetector::ghostSuppressFilter(const cv::Mat& mask){
    cv::Mat result=mask.clone();

    // 计算每像素光流幅值
    cv::Mat flowMag;
    cv::magnitude(cachedFlowX,cachedFlowY,flowMag);

    // 连通域分析
    cv::Mat labels,stats,centroids;
    int nc=cv::connectedComponentsWithStats(mask,labels,stats,centroids,8,CV_32S);

    // 对每个前景连通域检查是否"静止"(鬼影)
    for(int i=1;i<nc;i++){
        int a=stats.at<int>(i,cv::CC_STAT_AREA);
        if(a<p.minArea) continue;

        int left=stats.at<int>(i,cv::CC_STAT_LEFT);
        int top=stats.at<int>(i,cv::CC_STAT_TOP);
        int w=stats.at<int>(i,cv::CC_STAT_WIDTH);
        int h=stats.at<int>(i,cv::CC_STAT_HEIGHT);
        cv::Rect roi(left,top,w,h);

        // 计算该blob内平均光流幅值
        float sumFlow=0; int cnt=0;
        for(int rr=roi.y;rr<roi.y+roi.height&&rr<rows;rr++){
            const float* fmr=flowMag.ptr<float>(rr);
            for(int cc=roi.x;cc<roi.x+roi.width&&cc<cols;cc++){
                if(labels.at<int>(rr,cc)==i){ sumFlow+=fmr[cc]; cnt++; }
            }
        }
        float avgFlow=(cnt>0)?sumFlow/cnt:0;

        // 鬼影判定: 平均光流低 → 静止前景
        if(avgFlow<p.ghostFlowThresh){
            // 更新 ghostTimer
            for(int rr=roi.y;rr<roi.y+roi.height&&rr<rows;rr++){
                int* gtr=ghostTimer.ptr<int>(rr); uchar* rr_=result.ptr<uchar>(rr);
                for(int cc=roi.x;cc<roi.x+roi.width&&cc<cols;cc++){
                    if(labels.at<int>(rr,cc)==i){
                        gtr[cc]++;
                        if(gtr[cc]>p.ghostMaxFrames) rr_[cc]=0; // 持续静止→强制BG
                    }
                }
            }
        } else {
            // 运动目标: 重置ghostTimer
            for(int rr=roi.y;rr<roi.y+roi.height&&rr<rows;rr++){
                int* gtr=ghostTimer.ptr<int>(rr);
                for(int cc=roi.x;cc<roi.x+roi.width&&cc<cols;cc++){
                    if(labels.at<int>(rr,cc)==i) gtr[cc]=0;
                }
            }
        }
    }

    // 非前景区的ghostTimer衰减
    for(int r=0;r<rows;r++){ const uchar*mr=mask.ptr<uchar>(r); int* gtr=ghostTimer.ptr<int>(r);
        for(int c=0;c<cols;c++){
            if(mr[c]==0) gtr[c]=std::max(0,gtr[c]-1);
        }
    }

    return result;
}

cv::Mat MotionDetector::flowConsistencyFilter(const cv::Mat& mask){
    // 光流方向局部一致性: 随机方向(树叶)→抑制, 一致方向(目标)→保留
    cv::Mat flowMag;
    cv::magnitude(cachedFlowX,cachedFlowY,flowMag);
    cv::Mat magSafe=flowMag.clone();
    magSafe.setTo(1.0f,flowMag<1.0f);
    cv::Mat sx, cx;
    cv::divide(cachedFlowX,magSafe,sx); // cos = fx/|f|
    cv::divide(cachedFlowY,magSafe,cx); // sin = fy/|f| (命名随意)
    cv::Mat meanSx, meanCx;
    cv::boxFilter(sx,meanSx,CV_32F,cv::Size(11,11));
    cv::boxFilter(cx,meanCx,CV_32F,cv::Size(11,11));
    cv::Mat consistency;
    cv::magnitude(meanSx,meanCx,consistency); // ∈ [0,1]

    cv::Mat result=mask.clone();
    for(int r=0;r<rows;r++){ const uchar*mr=mask.ptr<uchar>(r);
        const float* cs=consistency.ptr<float>(r); uchar*rr=result.ptr<uchar>(r);
        for(int c=0;c<cols;c++){
            if(mr[c]==255 && cs[c]<0.3f) rr[c]=0;
        }
    }
    return result;
}

cv::Mat MotionDetector::staticFilter(const cv::Mat& mask){
    // 帧差极小+纹理区域→不可能是真实运动→强制BG
    // 纹理区域(localStd高)的微小帧差来自压缩噪声, 平坦区域的帧差已由ViBe正确处理
    cv::Mat diff; cv::absdiff(grayFrame,previousGray,diff);
    cv::Mat stdNorm=computeLocalStdNorm(); // localStd归一化 [0,1]
    cv::Mat result=mask.clone();
    for(int r=0;r<rows;r++){ const uchar*mr=mask.ptr<uchar>(r),*dr=diff.ptr<uchar>(r);
        const float* sn=stdNorm.ptr<float>(r); uchar*rr=result.ptr<uchar>(r);
        for(int c=0;c<cols;c++){
            if(mr[c]==255 && dr[c]<p.staticThresh && sn[c]>0.7f) rr[c]=0;
        }
    }
    return result;
}

// ============================================================================
// 颜色精修：灰度ViBe判为FG但颜色接近BG → 疑似FP → 抑制
// 利用颜色信息提升Precision，不产生新FN (真FG颜色必然异于BG)
// ============================================================================
cv::Mat MotionDetector::colorRefine(const cv::Mat& mask){
    cv::Mat result = mask.clone();
    const int minDist = p.colorMinDist;
    for(int r=0;r<rows;r++){ const uchar* mr=mask.ptr<uchar>(r);
        const cv::Vec3b* cr_=colorFrame.ptr<cv::Vec3b>(r);
        const cv::Vec3b* br=bgDisplay.ptr<cv::Vec3b>(r);
        uchar* rr=result.ptr<uchar>(r);
        for(int c=0;c<cols;c++){
            if(mr[c]==255){
                cv::Vec3b fg=cr_[c], bg=br[c];
                int d=std::abs((int)fg[0]-(int)bg[0])
                     +std::abs((int)fg[1]-(int)bg[1])
                     +std::abs((int)fg[2]-(int)bg[2]);
                if(d < minDist) rr[c]=0; // 颜色太接近BG → FP
            }
        }
    }
    return result;
}

// ============================================================================
// 边缘吸附：将mask边界收缩到原图Canny边缘，消除边界漂移FP
// ============================================================================
cv::Mat MotionDetector::edgeSnap(const cv::Mat& mask, const cv::Mat& colorFrame){
    // 1. 原图Canny边缘
    cv::Mat rawGray;
    if(colorFrame.channels()==3) cv::cvtColor(colorFrame,rawGray,cv::COLOR_BGR2GRAY);
    else rawGray=colorFrame;
    cv::Mat edges;
    cv::Canny(rawGray,edges,40,120,3);
    cv::dilate(edges,edges,cv::Mat(),cv::Point(-1,-1),1);

    // 2. mask边界 = xor(mask, eroded_mask)
    cv::Mat eroded, boundary;
    cv::erode(mask,eroded,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));
    cv::bitwise_xor(mask,eroded,boundary);

    // 3. 边界像素中, 不在边缘上的 → 属于漂移 → 去除
    cv::Mat drift; // 边界但不在edge上
    cv::bitwise_and(boundary, cv::Scalar(255)-edges, drift);
    cv::Mat result = mask.clone();
    result.setTo(0, drift);

    return result;
}

// ============================================================================
// 轮廓填充：提取外部轮廓并填充，消除内部空洞
// 在 Guided Filter 已对齐边缘后执行，不会跨越语义边界
// ============================================================================
cv::Mat MotionDetector::fillHoles(const cv::Mat& mask){
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    cv::Mat result = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(result, contours, -1, cv::Scalar(255), cv::FILLED);
    return result;
}

// ============================================================================
cv::Mat MotionDetector::filterSmallComponents(const cv::Mat& mask,
                                               const cv::Mat& motionMap){
    cv::Mat labels,stats,centroids;
    int nc=cv::connectedComponentsWithStats(mask,labels,stats,centroids,8,CV_32S);
    cv::Mat f(rows,cols,CV_8UC1,cv::Scalar(0));
    for(int i=1;i<nc;i++){
        int a=stats.at<int>(i,cv::CC_STAT_AREA);
        // 自适应面积阈值: 运动证据强的blob → 放宽阈值保留小目标
        int left=stats.at<int>(i,cv::CC_STAT_LEFT);
        int top=stats.at<int>(i,cv::CC_STAT_TOP);
        int w=stats.at<int>(i,cv::CC_STAT_WIDTH);
        int h=stats.at<int>(i,cv::CC_STAT_HEIGHT);
        // 计算blob内平均运动证据
        float motionSum=0; int motionCnt=0;
        cv::Rect roi(std::max(left,0),std::max(top,0),
                     std::min(w,cols-left),std::min(h,rows-top));
        if(roi.width>0&&roi.height>0){
            cv::Mat blobMask=labels(roi)==i;
            for(int rr=0;rr<roi.height;rr++){
                const uchar* bmr=blobMask.ptr<uchar>(rr);
                const float* mmr=motionMap.ptr<float>(top+rr)+left;
                for(int cc=0;cc<roi.width;cc++){
                    if(bmr[cc]){ motionSum+=mmr[cc]; motionCnt++; }
                }
            }
        }
        float avgMotion=(motionCnt>0)?motionSum/motionCnt:0;
        // 运动强的小目标: 阈值降至 minArea/3
        int effMinA=p.minArea;
        if(avgMotion>0.3f) effMinA=p.minArea/3;
        if(a<effMinA)continue;
        cv::Mat mi=(labels==i); f.setTo(255,mi);
    }
    cv::Mat k=cv::getStructuringElement(cv::MORPH_ELLIPSE,
                cv::Size(p.morphKernelSize,p.morphKernelSize));
    cv::Mat t;
    cv::morphologyEx(f,t,cv::MORPH_CLOSE,k);
    cv::morphologyEx(t,f,cv::MORPH_OPEN,k);
    return f;
}

// ============================================================================
void MotionDetector::buildOverlay(const cv::Mat& mask){
    rawFrame.copyTo(overlay);
    for(int r=0;r<rows;r++){ const uchar* mr=mask.ptr<uchar>(r); cv::Vec3b* orr=overlay.ptr<cv::Vec3b>(r);
        for(int c=0;c<cols;c++)if(mr[c]==255){
            orr[c][0]=(uchar)(orr[c][0]*0.4);orr[c][1]=(uchar)(orr[c][1]*0.4+255*0.6);orr[c][2]=(uchar)(orr[c][2]*0.4);
        }
    }
    std::vector<std::vector<cv::Point>> ct;
    cv::findContours(mask.clone(),ct,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    for(auto& cc:ct)if(cv::contourArea(cc)>=p.minArea)
        cv::drawContours(overlay,std::vector<std::vector<cv::Point>>{cc},-1,cv::Scalar(0,255,0),2);
}

cv::Mat MotionDetector::getForeground() const {
    if(currentMask.empty() || rawFrame.empty()) return cv::Mat();
    cv::Mat fg;
    cv::bitwise_and(rawFrame, rawFrame, fg, currentMask);
    return fg;
}

double MotionDetector::median(const cv::Mat& mat){
    cv::Mat s; cv::sort(mat,s,cv::SORT_EVERY_ROW+cv::SORT_ASCENDING);
    int n=mat.cols; if(n%2)return s.at<float>(0,n/2);
    else return 0.5f*(s.at<float>(0,n/2-1)+s.at<float>(0,n/2));
}
