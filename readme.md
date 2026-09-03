"""
Outputs / 输出文件说明
=====================


INPUT / 输入图像
----------------

00_rgb_original_gt.jpg
    原始 RGB 图像 + UAV 真实标注框。
    用于观察无人机在原始高分辨率可见光图像中的位置、尺度及背景环境。


01_tir_original_gt.jpg
    原始 TIR 热红外图像 + UAV 真实标注框。
    用于观察无人机在原始热红外图像中的位置及热响应特征。


02_rgb_semantic_gt.jpg
    RGB 语义分支输入图像 + UAV 真实标注框。
    即原始 RGB 经 LetterBox / Resize 后得到的低分辨率语义输入，
    实际送入 RGB Backbone。


03_tir_letterbox_gt.jpg
    TIR 经 LetterBox 后的网络输入 + UAV 真实标注框。
    实际送入 TIR Backbone。


============================================================
RLSFA ALIGNMENT / RLSFA 可靠性引导局部空频对齐
仅保留关键对齐过程
============================================================


04_rgb_tir_p3_before_alignment.jpg
    RLSFA 对齐前的 RGB P3 与原始 TIR P3 特征对比。

    左侧：
        RGB P3 特征响应。

    右侧：
        未对齐 TIR P3 特征响应。

    主要用于观察：
        RGB 与 TIR 中 UAV 特征响应中心是否存在明显空间错位。


05_coarse_targetness.jpg
    Coarse Matcher 的目标支持图（Targetness Map）。

    该图表示：
        在 RGB 参考坐标系中，哪些位置更可能属于 UAV 目标区域。

    作用：
        在粗粒度相关性搜索时，提高 UAV 区域的权重，
        降低建筑、天空、植被等大面积背景对跨模态匹配的干扰。

    注意：
        该图不是 UAV 分割结果。


06_tir_after_coarse.jpg
    Coarse Translation 粗平移后的 TIR 结果。

    左侧：
        原始 TIR。

    右侧：
        根据 coarse_offset 平移后的 TIR。

    用于观察：
        粗对齐是否已经将 TIR 中 UAV 的位置移动到 RGB 目标附近。


07_local_spatial_frequency_panel.jpg
    TIR 局部空频可靠结构四宫格。

    四个子图的背景均为：
        Coarse-aligned TIR
        即经过粗平移后的 TIR。

    四个子图依次为：

    1. TIR spatial structure
       TIR 空间域结构响应。

       由多尺度可学习卷积分支提取，
       主要用于提供稳定、明确的局部空间位置和结构信息。


    2. TIR local Window-FFT
       TIR 局部窗口频率响应。

       通过多尺度重叠 Window FFT 获得，
       用于补充微小 UAV 的局部高频和细节结构信息。

       注意：
           高频响应不等于 UAV 边界，
           建筑边缘、纹理、热对比变化也可能产生高频响应。


    3. Frequency reliability Q_T
       TIR 频率信息可靠性图。

       Q_T 越接近 1：
           当前位置更加信任局部频率信息。

       Q_T 越接近 0：
           当前位置更加信任空间结构信息。


    4. Reliable TIR structure
       可靠性选择后的最终 TIR 结构表征。

       计算方式：

           B_T = Q_T * F_T + (1 - Q_T) * S_T

       其中：
           F_T = TIR 局部频率结构
           S_T = TIR 空间结构
           Q_T = TIR 频率可靠性

       用于避免不可靠的频率响应直接干扰后续 Fine Matching。


08_fine_displacement_probability.jpg
    Fine Matcher 的候选残差位移概率分布图。

    该图不再是传统空间热图，
    而是直接展示 Fine Correlation 实际产生的候选位移概率。

    当前：

        fine_radius = 2

    因此候选位移范围为：

        dx ∈ {-2, -1, 0, +1, +2}
        dy ∈ {-2, -1, 0, +1, +2}

    共：

        5 × 5 = 25

    个候选残差平移。

    横轴：
        dx

    纵轴：
        dy

    每个格子的颜色和数值表示：
        当前候选位移的 Softmax probability。

    概率越大：
        Fine Matcher 越倾向于采用该残差位移。

    图中同时显示：

        fine_offset
        fine_confidence

    其中：

        fine_offset
            为最终计算得到的连续残差平移量。

        fine_confidence
            表示 Fine Matcher 对当前位移搜索结果的整体置信程度。

    该图用于判断：
        Fine Matcher 是否形成明确的位移概率峰值，
        以及是否只对 Coarse Alignment 进行小范围修正。


09_total_offset.jpg
    RLSFA 最终总平移量。

    计算方式：

        total_offset
        =
        coarse_offset
        +
        fine_offset

    图中箭头表示：
        最终 TIR 相对于 RGB 参考坐标系需要进行的平移方向和幅度。

    注意：
        RLSFA 只执行全局平移，
        不执行仿射缩放，
        不执行稠密非刚性形变。


10_rgb_tir_after_final_rlsfa.jpg
    RLSFA 最终对齐结果。

    用于综合比较：

        RGB reference
        Aligned TIR
        Aligned TIR P3 response

    主要观察：
        最终 TIR 中 UAV 是否已经移动到 RGB UAV 对应位置附近，
        同时是否保持原有目标尺寸和结构。


============================================================
BHLR HIGH-RESOLUTION LOST-DETAIL RECOVERY
BHLR 高分辨率丢失细节恢复
============================================================


11_rgb_high_gt.jpg
    高分辨率 RGB 输入 + UAV 真实标注框。

    这是 BHLR 的高分辨率信息来源。

    例如：

        RGB-high = 1280 × 1280
        或
        RGB-high = 1920 × 1920


12_rgb_semantic_upscaled.jpg
    RGB semantic 重新上采样到 RGB-high 分辨率后的结果。

    表示：
        如果只保留低分辨率 RGB semantic，
        再通过普通插值恢复到高分辨率，
        最多能够恢复出什么信息。

    它与 RGB-high 的差异，
    就是后续 BHLR 要利用的“降采样丢失信息”。


13_highres_residual.jpg
    高分辨率丢失细节残差图。

    计算方式：

        Residual
        =
        RGB_high
        -
        Up(RGB_semantic)

    表示：
        RGB 从高分辨率降采样到 semantic 分辨率过程中，
        未被低分辨率输入保留下来的细节信息。

    其中可能包括：

        UAV 轮廓
        UAV 局部纹理
        小目标边缘
        建筑边缘
        树叶纹理
        其他背景高频信息

    因此：
        Residual 本身并不全部是有用 UAV 信息。


14_highres_residual_overlay.jpg
    高分辨率残差热图叠加到 RGB-high 图像上的结果。

    用于直接观察：
        丢失的高分辨率细节主要分布在哪些位置。

    重点观察：
        UAV 区域是否存在明显 residual，
        同时背景中是否也存在大量高频残差信息。


15_uav_residual_zoom.jpg
    UAV 区域的高分辨率残差局部放大图。

    用于重点比较 UAV 局部区域中的：

        RGB-high
        Up(RGB-semantic)
        High-resolution Residual

    主要回答：

        高分辨率 RGB 相比低分辨率 semantic RGB，
        在微小 UAV 区域到底多保留了哪些信息？


16_pixel_unshuffle_detail.jpg
    PixelUnshuffle 后的高分辨率细节表示。

    BHLR 不直接在高分辨率空间中处理全部 residual，
    而是通过 PixelUnshuffle 将高分辨率空间信息搬移到通道维。

    例如 ratio = 2：

        [B, 3, 1280, 1280]
        ->
        [B, 12, 640, 640]

    ratio = 3：

        [B, 3, 1920, 1920]
        ->
        [B, 27, 640, 640]

    这样可以：
        保留局部高分辨率子像素信息，
        同时将空间尺寸恢复到 semantic RGB 尺度。


17_detail_encoder_output.jpg
    Shared Lost-detail Encoder 输出。

    PixelUnshuffle 后的 residual 特征经过：

        Conv
        Depthwise Conv
        1×1 Conv

    得到共享高分辨率细节特征库：

        detail_raw

    它包含：
        后续各尺度可以使用的高分辨率丢失信息。


18_detail_avg_pool_p3.jpg
    detail_raw 经 Adaptive Average Pooling 后得到的 P3 尺度特征。

    主要表示：
        区域内相对稳定、平均的高分辨率细节响应。


19_detail_max_pool_p3.jpg
    detail_raw 经 Adaptive Max Pooling 后得到的 P3 尺度特征。

    主要表示：
        区域内最强的局部高分辨率细节响应。


20_detail_projection_p3.jpg
    P3 尺度最终高分辨率细节特征。

    计算过程：

        AvgPool(detail_raw)
                |
        MaxPool(detail_raw)
                |
             Concatenate
                |
             1×1 Conv
                |
             Detail_P3

    该特征已经：
        与 RGB P3 空间尺度和通道数匹配，

    是后续真正参与 BHLR 恢复的高分辨率细节表示。


21_support_map.jpg
    BHLR Soft UAV Support Map。

    由：

        RGB semantic feature
        aligned TIR feature
        RGB/TIR structural information

    共同生成。

    表示：
        哪些区域更可能属于需要进行高分辨率细节恢复的 UAV 区域。

    注意：
        Support Map 不是语义分割结果，
        其作用是定义“高分辨率细节应该在哪里被使用”。


22_support_overlay.jpg
    Support Map 叠加在 RGB 图像上的结果。

    用于观察：
        BHLR 是否能够将高分辨率细节检索区域集中到 UAV 周围，

    而不是：
        对整幅图像无差别恢复高频信息。


23_support_uav_zoom.jpg
    UAV 区域 Support Map 局部放大图。

    用于重点观察：
        对于远距离微小 UAV，
        Support Map 是否能够覆盖完整 UAV 区域。


24_detail_gate.jpg
    高分辨率细节有效性门控图。

    Gate 用于判断：

        已经提取出来的高分辨率丢失细节，
        哪些真正对当前 RGB-T 检测有用。

    Gate 越接近 1：
        越倾向于保留该区域细节。

    Gate 越接近 0：
        越倾向于抑制该区域细节。


25_detail_gate_overlay.jpg
    Detail Gate 叠加到 RGB 图像上的结果。

    用于观察：
        Gate 是否主要保留 UAV 区域的有效细节，

    同时抑制：
        建筑纹理
        树叶
        地面纹理
        其他无关背景高频信息。


26_usable_detail.jpg
    最终可用于恢复的高分辨率细节。

    计算方式：

        Usable Detail
        =
        Support
        ×
        Gate
        ×
        Detail_P3

    即：

        D_use = M × G × D

    这是 BHLR 中最关键的中间变量之一。

    它表示：
        最终真正允许注入 RGB P3 的高分辨率丢失信息。


27_usable_detail_overlay.jpg
    Usable Detail 叠加在 RGB 图像上的结果。

    用于判断：

        原始 residual 中大量背景高频信息，
        是否经过 Support + Gate 后被有效过滤，

    最终有效细节是否主要集中在 UAV 附近。


28_rgb_before_after_bhlr.jpg
    BHLR 增强前后 RGB P3 特征对比。

    左侧：
        RGB P3 before BHLR

    右侧：
        RGB P3 after BHLR

    用于观察：
        注入高分辨率丢失细节后，
        UAV 区域的特征响应是否增强。


29_bhlr_enhancement.jpg
    BHLR 实际注入到 RGB P3 中的特征增量。

    计算方式：

        Enhancement
        =
        RGB_after
        -
        RGB_before

    根据当前 BHLR：

        RGB_after
        =
        RGB_before
        +
        tanh(gamma) * usable_detail

    因此：

        Enhancement
        =
        tanh(gamma) * usable_detail

    该图表示：
        最终真正改变 RGB 检测特征的高分辨率信息。


============================================================
DETECTION / 最终检测
============================================================


30_prediction_vs_gt.jpg
    最终检测结果与真实标注对比。

    一般：

        红框 = Ground Truth
        绿框 = Prediction

    用于观察：

        RLSFA 对齐
        +
        BHLR 高分辨率细节恢复

    最终是否能够改善 UAV 检测结果。


============================================================
OTHER OUTPUTS / 其他输出
============================================================


overview.jpg
    所有关键中间可视化结果的汇总图。

    用于：
        快速浏览当前样本从输入、
        对齐、
        高分辨率细节恢复，
        到最终检测的完整过程。


metadata.json
    当前样本对应的数值型调试信息。

    主要包括：

        输入图像尺寸
        P3 特征尺寸

        coarse_offset
        fine_offset
        total_offset

        fine_confidence
        fine_probability_max
        fine_argmax_dx
        fine_argmax_dy

        RGB frequency reliability mean
        TIR frequency reliability mean

        Targetness mean

        BHLR support mean
        Detail gate mean
        Gamma

    用于对可视化结果进行定量辅助分析。
"""