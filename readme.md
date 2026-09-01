Outputs / 输出文件说明
---------------------
INPUT / 输入图像
00_rgb_original_gt.jpg 原始 RGB 图像 + UAV 真实标注框
01_tir_original_gt.jpg 原始 TIR 热红外图像 + UAV 真实标注框
02_rgb_semantic_gt.jpgRGB 
    语义分支输入图像 + UAV 真实标注框 
    即送入 RGB Backbone 的低分辨率 RGB 输入
03_tir_letterbox_gt.jpg TIR 
    经 LetterBox 后的网络输入 + UAV 
    真实标注框 RLSFA ALIGNMENT / RLSFA 对齐过程
04_rgb_tir_p3_before_alignment.jpg
    RLSFA 对齐前的 RGB P3 与 TIR P3 特征对比
    用于观察两种模态目标响应在空间位置上的错位情况
05_coarse_targetness.jpg
    粗对齐阶段的目标支持图 / Targetness Map
    表示 Coarse Matcher 更关注哪些可能属于 UAV 的区域，
    用于抑制大面积背景对跨模态相关性搜索的干扰
06_tir_after_coarse.jpg
    粗平移对齐后的 TIR 结果
    用于观察 Coarse Translation 是否已经将 TIR 目标移向 RGB 参考位置
07_local_spatial_frequency_panel.jpg
    TIR 局部空频结构可靠性四宫格
    包括：
        1. 空间结构响应
        2. 局部 Window-FFT 频率响应
        3. 频率可靠性 Q_T
        4. 最终可靠 TIR 结构
    用于观察空间域与频率域信息如何进行可靠性自适应选择
08_fine_correlation.jpg
    精细对齐阶段的局部跨模态相关性响应
    用于观察粗对齐后 UAV 附近是否形成更明确的局部对应关系
09_total_offset.jpg
    RLSFA 最终总平移量
    total_offset = coarse_offset + fine_offset
    仅进行全局平移，不执行仿射变换或稠密非刚性形变
10_rgb_tir_after_final_rlsfa.jpg
    RLSFA 最终对齐结果
    用于比较 RGB 参考目标与最终对齐后的 TIR 目标空间对应情况
BHLR HIGH-RESOLUTION LOST-DETAIL RECOVERY
BHLR 高分辨率丢失细节恢复
11_rgb_high_gt.jpg
    高分辨率 RGB 输入 + UAV 真实标注框
    表示 BHLR 使用的原始高分辨率细节信息来源
12_rgb_semantic_upscaled.jpg
    RGB 语义输入重新上采样到高分辨率后的结果
    表示仅依赖低分辨率 RGB 时能够恢复出的图像信息
13_highres_residual.jpg
    高分辨率丢失细节残差图
    residual = RGB_high - Up(RGB_semantic)
    表示 RGB 降采样过程中未被低分辨率语义输入保留下来的信息
14_highres_residual_overlay.jpg
    高分辨率残差热图叠加在 RGB-high 上的结果
    用于直观观察丢失细节主要分布在 UAV、背景纹理还是物体边缘区域
15_uav_residual_zoom.jpg
    UAV 区域高分辨率残差局部放大图
    用于直接比较 UAV 在高分辨率 RGB、低分辨率重建结果以及残差中的细节差异
16_pixel_unshuffle_detail.jpg
    PixelUnshuffle 后的高分辨率细节表示
    将高分辨率空间细节重新排列到通道维度，
    在降低空间尺寸的同时保留局部高分辨率信息
17_detail_encoder_output.jpg
    高分辨率细节编码器输出
    表示经过 Shared Lost-detail Encoder 后形成的共享高分辨率细节特征库
18_detail_avg_pool_p3.jpg
    高分辨率细节特征经过 Average Pooling 后得到的 P3 尺度表示
    更强调区域平均和稳定的细节响应
19_detail_max_pool_p3.jpg
    高分辨率细节特征经过 Max Pooling 后得到的 P3 尺度表示
    更强调局部最强的高分辨率细节响应
20_detail_projection_p3.jpg
    AvgPool 与 MaxPool 特征融合并投影后的 P3 高分辨率细节特征
    这是后续真正参与 BHLR 恢复的尺度匹配细节表示
21_support_map.jpg
    BHLR 软目标支持图
    由 RGB/TIR 语义与结构信息共同生成，
    表示哪些区域更可能属于需要进行高分辨率细节恢复的 UAV 区域
    注意：该图不是目标分割掩膜
22_support_overlay.jpg
    BHLR 支持区域叠加图
    将 Support Map 叠加到 RGB 图像上，
    用于观察高分辨率细节检索区域是否集中在 UAV 附近
23_support_uav_zoom.jpg
    UAV 区域 Support Map 局部放大图
    用于观察微小 UAV 区域是否被有效覆盖
24_detail_gate.jpg
    高分辨率细节有效性门控图
    用于判断候选丢失细节中哪些信息值得保留并注入 RGB 特征

25_detail_gate_overlay.jpg
    Detail Gate 叠加到 RGB 图像后的结果
    用于观察门控是否主要选择 UAV 相关细节而抑制背景细节
26_usable_detail.jpg
    最终可用高分辨率细节特征
    usable_detail = support × gate × detail
    表示真正允许注入 RGB P3 的高分辨率丢失信息
27_usable_detail_overlay.jpg
    最终可用细节在 RGB 图像上的叠加结果
    用于观察 BHLR 是否将高分辨率信息集中恢复到 UAV 有效区域
28_rgb_before_after_bhlr.jpg
    BHLR 增强前后 RGB P3 特征对比
    左侧为原始 RGB P3，
    右侧为注入高分辨率丢失细节后的 RGB P3
29_bhlr_enhancement.jpg
    BHLR 实际特征增强量
    enhancement = RGB_after - RGB_before
    表示经过 gamma 调制后真正加入 RGB 检测特征中的信息
DETECTION / 最终检测结果
30_prediction_vs_gt.jpg
    最终检测结果与真实标注对比
    用于观察 RLSFA + BHLR 最终对 UAV 检测结果的影响
OTHER OUTPUTS / 其他输出
overview.jpg
    所有关键中间结果的汇总预览图
metadata.json
    当前样本的可视化元数据
    包括输入尺寸、特征尺寸、Coarse/Fine/Total Offset、
    可靠性均值、Support/Gate/Gamma 等数值信息