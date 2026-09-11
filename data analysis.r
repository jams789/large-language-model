
dat <- m

models <- c("Qwen3","Deepseek","GPT4","Medgemma")

sapply(models, function(m){
  x <- sum(dat[[m]] == dat$truth)
  n <- nrow(dat)
  ci <- binom.test(x, n)$conf.int
  sprintf("%.1f%% (%.1f%%–%.1f%%)", x/n*100, ci[1]*100, ci[2]*100)
})

# 假设数据已读入 df
# df <- read.table("your_data.txt", header=TRUE)
df <-dat

data <-dat
# 配对McNemar函数

#
mcnemar_pair <- function(m1, m2, truth) {
  c1 <- m1 == truth
  c2 <- m2 == truth
  
  b <- sum(c1 & !c2)   # model1 对，model2 错
  c <- sum(!c1 & c2)   # model1 错，model2 对
  n <- length(truth)
  
  tab <- matrix(c(
    sum(c1 & c2),
    b,
    c,
    sum(!c1 & !c2)
  ), nrow = 2, byrow = TRUE)
  
  p <- mcnemar.test(tab, correct = FALSE)$p.value
  
  # paired accuracy difference = (b - c) / n
  se <- sqrt(b + c) / n
  z  <- ((b - c) / n) / se
  
  data.frame(
    p_value = p,
    Standard_Error = se,
    Z_Score = z
  )
}

res <- data.frame()

for (i in 1:(length(models)-1)) {
  for (j in (i+1):length(models)) {
    
    tmp <- mcnemar_pair(
      df[[models[i]]],
      df[[models[j]]],
      df$truth
    )
    
    res <- rbind(res, data.frame(
      model1 = models[i],
      model2 = models[j],
      tmp
    ))
  }
}

res$p_BH <- p.adjust(res$p_value, method = "BH")

res

models <- c("Qwen3","Deepseek","GPT4","sr","jr","Medgemma")

scores <- 1:5

result <- lapply(models, function(m) {
  sapply(scores, function(s) {
    idx <- dat$truth == s
    if (sum(idx) == 0) return(NA)
    mean(dat[[m]][idx] == s)
  })
})

result <- do.call(rbind, result)
colnames(result) <- paste0("score_", scores)
rownames(result) <- models

round(result, 4)
library(irr)
library(psych)

library(irr)

library(irr)

kappa_result <- lapply(models, function(m) {
  
  tmp <- data.frame(
    truth = dat$truth,
    pred  = dat[[m]]
  )
  
  tmp <- na.omit(tmp)
  
  res <- kappa2(tmp, weight = "weighted")
  
  k  <- res$value
  se <- k / res$statistic
  
  lower <- k - 1.96 * se
  upper <- k + 1.96 * se
  
  data.frame(
    Model = m,
    Kappa = k,
    CI_lower = max(-1, lower),
    CI_upper = min(1, upper)
  )
})

kappa_result <- do.call(rbind, kappa_result)
kappa_result



# 假设你的数据如下
truth  <- llm$truth
rater1 <- llm$jr
rater2 <- llm$Medgemma

# 计算每位评分者的正确情况
correct1 <- rater1 == truth
correct2 <- rater2 == truth

# 构造列联表
tbl <- table(correct1, correct2)
tbl
# 输出例子：
#           correct2
# correct1  FALSE TRUE
#     FALSE   1     2
#     TRUE    3     4

# McNemar 检验
mcnemar.test(tbl)










# ===== 必要包（首次使用请取消注释安装） =====
# install.packages(c("readxl","dplyr","ggplot2","irrCAC","scales"))
library(readxl)
library(dplyr)
library(ggplot2)
library(scales)
library(ggplot2)
library(ggfx)
library(scales)
library(viridis)
library(ggplot2)
library(scales)
library(viridis)
# ===== 1) 读取 Excel =====
# 改成你的文件路径、Sheet 名与列名
excel_path <- "C:\\Users\\Administrator\\Desktop\\大语言模型投稿\\rating.xlsx"
sheet_name <- 1               # 或 "Sheet1"
col_true   <- "truth"        # 例如 "gt" / "doctor" / "y_true"
col_pred   <- "original report"        # 例如 "pred" / "model" / "y_pred"

df <- read_excel(excel_path, sheet = sheet_name) |>
  rename(y_true = all_of(col_true),
         y_pred = all_of(col_pred)) |>
  # 去掉缺失
  filter(!is.na(y_true), !is.na(y_pred))

levels_scores <- 1:5


df <- df |>
  mutate(
    y_true = factor(y_true, levels = levels_scores, ordered = TRUE),
    y_pred = factor(y_pred, levels = levels_scores, ordered = TRUE)
  )

# ===== 3) 混淆矩阵与指标 =====
tab <- table(`truth` = df$y_true, `original report` = df$y_pred)
n   <- sum(tab)
acc <- sum(diag(tab)) / n

title_main <- sprintf("热力混淆矩阵")

# ===== 4) 计数热图 =====
tab_df <- as.data.frame(tab)
# y 轴从“1级”在上到“5级”在下
tab_df$truth <- factor(tab_df$truth, levels = rev(levels_scores), ordered = TRUE)
tab_df$original.report <- factor(tab_df$original.report, levels = levels_scores, ordered = TRUE)
# 需要的包
# install.packages(c("ggplot2","ggfx","scales","viridis"))
p_grad <- ggplot(tab_df, aes(x = truth, y = original.report, fill = Freq)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = Freq), size = 4, color = "black") +
  scale_fill_gradientn(
    colors = c("white", "#E6D3A0", "#C2900F"),
    name = "Count"
  ) +
  labs(title = title_main,
       x = "VI-RADS (origianl reports)", y = "VI-RADS (original.report)") +
  coord_fixed() +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, lineheight = 1.15, face = "bold"),
    legend.title = element_text(size = 12),
    legend.text  = element_text(size = 10)
  )

print(p_grad)
"#F5E7BC", "#C0845C"
"#C2C5A4", "#656D14"
"#F29091", "#D3272B"
"#FFBC80", "#F98F34"
"#B4DEA2", "#6BBC46"
"#6B98C4", "#0C4E9B"
"#E6D3A0", "#C2900F"




# 如需保存图片：
# ggsave("混淆矩阵_计数.png", p_count, width = 8, height = 8, dpi = 300)

# ===== 5)（可选）行归一百分比热图（每行和为 100%）=====
row_df <- prop.table(tab, margin = 1) |>
  as.data.frame() |>
  rename(Prop = Freq) |>
  mutate(Percent = sprintf("%.1f%%", Prop * 100))

row_df$`医师评分` <- factor(row_df$`医师评分`, levels = rev(levels_scores), ordered = TRUE)
row_df$`LLM评分` <- factor(row_df$`LLM评分`, levels = levels_scores, ordered = TRUE)

p_row <- ggplot(row_df, aes(x = `LLM评分`, y = `医师评分`, fill = Prop)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = Percent), size = 4) +
  scale_fill_gradient(low = "#f7fbff", high = "#08306b", labels = percent) +
  labs(title = sprintf("热力混淆矩阵（行归一占比）\nn=%d", n),
       x = "LLM评分", y = "医师评分", fill = "占比") +
  coord_fixed() +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, lineheight = 1.15, face = "bold"))
# print(p_row)
# ggsave("混淆矩阵_行归一.png", p_row, width = 8, height = 8, dpi = 300)









library(ggplot2)
library(dplyr)
library(patchwork)

plot2 <- ggplot(duibi, aes(x = llm, y = k, fill = llm)) +  
  geom_col(alpha = 0.8) +  # 画柱状图  
  scale_fill_manual(values = c("#274753", "#299d8f", "#8ab07c", "#e7c66b", "#e66d50", "#e66d50")) + 
  labs(title = "不同树种平均胸径比较", 
     x = "树种", y = "平均胸径 (cm)") +  
  theme_minimal() +  
  theme(legend.position = "none")  # 不显示图例
  print(plot2)
  
  
  
  
  
  
  plot2 <- ggplot(duibi, aes(x = reorder(llm, k), y = k, fill = llm)) +
    geom_col(width = 0.7, alpha = 0.9, color = "white") +  # 调整宽度与边框
    scale_fill_manual(
      values = c("#274753", "#299d8f", "#8ab07c", "#e7c66b", "#e66d50", "#d94f30")
    ) +
    labs(
      title = "不同树种平均胸径比较",
      subtitle = "各树种平均胸径（cm）",
      x = "树种",
      y = "平均胸径 (cm)"
    ) +
    theme_minimal(base_size = 14, base_family = "STHeiti") +  # 设置中文友好字体
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray30"),
      axis.text.x = element_text(angle = 45, hjust = 1, color = "gray20"),
      axis.text.y = element_text(color = "gray20"),
      axis.title.x = element_text(vjust = -0.5),
      panel.grid.major.x = element_blank(),
      legend.position = "none",
      plot.margin = margin(10, 10, 10, 10)
    )
  
  print(plot2)  

  
  
  library(ggplot2)
  
  plot2 <- ggplot(duibi, aes(x = reorder(llm, k), y = k, fill = llm)) +
    geom_col(width = 0.7, alpha = 0.9, color = "white") +  # 柱状图主体
    geom_text(aes(label = round(k, 2)),                    # 添加数值标签
              vjust = -0.5, size = 4, color = "black", fontface = "bold") +
    scale_fill_manual(
      values = c("#274753", "#299d8f", "#8ab07c", "#e7c66b", "#e66d50", "#d94f30")
    ) +
    labs(
      title = "不同树种平均胸径比较",
      subtitle = "各树种平均胸径（cm）",
      x = " ",
      y = "平均胸径 (cm)"
    ) +
    theme_minimal(base_size = 14, base_family = "STHeiti") +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray30"),
      axis.text.x = element_text(angle = 45, hjust = 1, color = "gray20"),
      axis.text.y = element_text(color = "gray20"),
      axis.title.x = element_text(vjust = -0.5),
      panel.grid.major.x = element_blank(),
      legend.position = "none",
      plot.margin = margin(10, 10, 10, 10)
    )
  
  print(plot2)  
  plot2 <- ggplot(duibi, aes(x = reorder(llm, k), y = k, fill = llm)) +
    geom_col(width = 0.7, alpha = 0.9, color = "white") +
    geom_text(
      aes(label = round(k, 2)),
      vjust = -0.5, size = 4, color = "black", fontface = "bold"
    ) +
    scale_fill_manual(
      values = c( "#91CDC8", "#6FB9D0","#386195", "#5499BD", "#3981AF", "#324C63")
    ) +
    scale_y_continuous(
      limits = c(0, max(duibi$k) * 1.15),   # 上限扩大15%
      expand = c(0, 0)
    ) +
    labs(
      title = "不同树种平均胸径比较",
      subtitle = "各树种平均胸径（cm）",
      x = "树种",
      y = "Agreement"
    ) +
    theme_minimal(base_size = 14, base_family = "STHeiti") +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray30"),
      axis.text.x = element_text(angle = 45, hjust = 1, color = "gray20"),
      axis.text.y = element_text(color = "gray20"),
      axis.title.x = element_text(vjust = -0.5),
      panel.grid.major.x = element_blank(),
      legend.position = "none",
      plot.margin = margin(10, 10, 10, 10)
    )
  
  print(plot2) 

#单组
 k 
  ggplot(long_data, aes(x = `样本`, y = Value, fill = `样本`)) +
    geom_col(
      width = 0.6,
      color = NA,
      linewidth = 0.05,
      alpha = 1
    ) +
    scale_fill_manual(
      values = c(
        "sr"       = "#B4E1F1",
        "jr"       = "#83B9BB",
        "Qwen3"    = "#BCB5D0",
        "MedGemma" = "#9384B4",
        "DeepSeek" = "#F2A2AF",
        "GPT-4.0"  = "#939597"
      )
    ) +
    theme_minimal() +
    coord_flip() +
    geom_text(
      aes(label = Value),
      vjust = 0.5,
      hjust = -0.2,
      size = 3.5,
      angle = 0,
      family = "serif",
      fontface = "plain",
      color = "black",
      check_overlap = TRUE
    ) +
    labs(x = "Y_Title", y = "X_Title") +
    scale_y_continuous(
      limits = c(0, 1.00),
      breaks = seq(0, 1.00, by = 0.1),
      labels = function(y) sprintf("%.0f", y),
      expand = c(0, 0)
    ) +
    theme_classic() +  # 这行会覆盖上面的 theme_minimal，二选一即可
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      axis.ticks.length = unit(0.1, "cm"),
      axis.title.x = element_text(family = "sans", size = 12, face = "italic",
                                  margin = margin(t = 5), color = "black"),
      axis.title.y = element_text(family = "sans", size = 12, face = "italic",
                                  angle = 90, margin = margin(r = 5), color = "black"),
      axis.text.x = element_text(family = "serif", size = 10, face = "bold.italic",
                                 margin = margin(t = 3), color = "black"),
      axis.text.y = element_text(family = "serif", size = 10, face = "bold.italic",
                                 margin = margin(r = 3), color = "black"),
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm")
    )
#纵向单组
  # 在绘图前重新定义因子水平顺序
  long_data$`样本` <- factor(long_data$`样本`, 
                           levels = c("Radiologists", "original report","Qwen3", "MedGemma", "GPT-4.0","DeepSeek" ))
  ggplot(long_data, aes(x = `样本`, y = k, fill = `样本`)) +
    geom_col(
      width = 0.40,
      color = NA,
      linewidth = 0.05,
      alpha = 1
    ) +
    geom_errorbar(
      aes(ymin = k_low, ymax = k_high),
      width = 0.15,
      linewidth = 0.6,
      color = "black"
    ) +
    scale_fill_manual(
      values = c(
        "Radiologists"  = "#354e97",
        "original report"  = "#F2A2AF",
        "Qwen3"    = "#c7e5ec",
        "MedGemma" = "#f5b46f",
        "DeepSeek" = "#a17db4",
        "GPT-4.0"  = "#df5b3f"
      )
    ) +
    # 移除 coord_flip() 并调整标签位置
    geom_text(
      aes(label = k),
      vjust = -0.2,  # 标签在柱子上方
      hjust = 0.5,   # 水平居中
      size = 5.0,
      angle = 0,
      family = "serif",
      fontface = "plain",
      color = "black",
      check_overlap = TRUE
    ) +
    labs(x = "X_Title", y = "Y_Title") +  # 交换横纵轴标题
    scale_y_continuous(
      limits = c(0, 1.00),
      breaks = seq(0, 1.00, by = 0.2),
      labels = function(y) sprintf("%.1f", y),  # 改为保留一位小数，因为范围是0-1
      expand = c(0, 0)
    ) +
    theme_classic() +
    theme(
      panel.grid.major.x = element_blank(),  # 移除X轴主网格线
      panel.grid.minor.x = element_blank(),  # 移除X轴次网格线
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.2),  # 添加Y轴网格线
      panel.grid.minor.y = element_blank(),
      axis.ticks.length = unit(0.1, "cm"),
      axis.title.x = element_text(
        family = "sans", 
        size = 16, 
        face = "italic",
        margin = margin(t = 5), 
        color = "black"
      ),
      axis.title.y = element_text(
        family = "sans", 
        size = 16, 
        face = "italic",
        margin = margin(r = 5),  # 移除 angle = 90
        color = "black"
      ),
      axis.text.x = element_text(
        family = "serif", 
        size = 16, 
        face = "bold.italic",
        margin = margin(t = 3), 
        color = "black"
      ),
      axis.text.y = element_text(
        family = "serif", 
        size = 16, 
        face = "bold.italic",
        margin = margin(r = 3), 
        color = "black"
      ),
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
      legend.position = "none"  # 如果不需要图例可以移除，因为颜色已经通过x轴标签区分
    )  
    
  
  library(readxl)
  # 用于读取数据
  library(ggplot2) 
  # 用于绘图
  library(tidyr) 
  # 用于数据格式转换
  library(showtext)
  # 用于支持中文字体# 添加中文字体支持（Windows系统）
  font_add("simsun","C:/Windows/Fonts/simsun.ttc")# 宋体
  font_add("simhei","C:/Windows/Fonts/simhei.ttf")# 黑体showtext_auto()
  #  启用字体渲染 
  y_max <- max(long_data$Value, na.rm = TRUE)
  y_min <- min(long_data$Value, na.rm = TRUE)

  long_data<-duibi
  
  
  
  # 3.绘制分组柱状图
  #---------------
  
  ggplot(long_data, aes(x= 样本,y= Value, fill = Index_group)) +
    # 3.1 绘制柱状图
    geom_col(position = position_dodge(0.75), # 分组间距(dodge表示分组并列放置，而非堆叠；数字表示组内柱子的间距，应等同于柱子宽度)           
             width =0.75, # 柱子宽度
             color ="NA", # 边框颜色(NA为无边框; "black"为黑色框线)
             linewidth =0.05, # 边框粗细
             alpha =1# 柱子透明度(0-1)
    ) +# 3.2 给柱子填充颜色
    scale_fill_manual(values= c('#354e97',"#F2A2AF",'#c7e5ec','#f5b46f','#df5b3f','#a17db4')
    )+
    # 3.3 翻转X轴和Y轴 
    coord_flip()+
    # 3.4 添加数值标签
    geom_text( 
      aes(label = Value),# 要显示的数值（映射long_data中的Value列）
      position = position_dodge(width =0.75), # 用于避免不同组别的标签重叠，避免宽度0.75（必须与geom_col的dodge宽度一致）
      vjust =0.5, # 垂直对齐：0（柱子左侧），0.5（柱子中间），1.5（柱子右侧）
      hjust = -0.2, # 水平对齐：0（标签在柱子上方紧贴柱子），1（标签在柱子上方紧贴柱子）
      size =3.5, # 字体大小（建议3-4）
      angle =0, # 旋转角度
      family ="serif", # 字体
      fontface ="plain", # 字体格式：常规字体(plain); 粗体(bold); 斜体(italic); 粗斜体(bold.italic)
      color ="black", # 字体颜色
      check_overlap = TRUE # 避免标签重叠
    )+
    # 3.5 修改标题 
    labs(x ="Y_Title", # 修改横轴标题
          y ="X_Title", # 修改纵轴标题
          fill ="Index")+# 修改图例标题
    # 3.6 精修Y轴
    scale_y_continuous(
      limits = c(0,100), # Y轴范围(Y轴从0绘制到0.4)
      breaks = seq(0, 100, by = 5), # 刻度(从0~0.40，步长为0.05 
      labels = function(y)sprintf("%.0f",y), # Y轴保留两位小数
      expand = c(0,0) # Y轴除了limits()规定的范围无额外扩展 
    )+
    #-------注意--------------------------------------------------# 如果要换自己的数据，拿不准坐标轴范围，可以换成
    # limits = c(0, y_max) 和 breaks = seq(0, y_max, by = 5)
    # 再根据绘制出的图调整精准的坐标轴范围(精准的坐标范围会图好看一点)
    #-------------------------------------------------------------
  # 3.7 设置主题
  theme_classic() +# 经典风格(仅保留坐标轴和刻度线)# 3.7.1 设置网格线和图例宽度 
    theme(
      panel.grid.major.y = element_blank(), # 移除Y轴主网格线 
      panel.grid.minor.y = element_blank(), # 移除Y轴次网格线
      panel.grid.major.x = element_blank(), # 移除X轴主网格线
      panel.grid.minor.x = element_blank(), # 移除X轴次网格线
      axis.ticks.length = unit(0.1,"cm"), # 刻度线朝向（正号朝外，负号朝内） 
      legend.key.width = unit(0.6,"cm"), # 图例键宽度
      # 3.7.2 横纵轴标题的细节
      axis.title.x = element_text(
        family ="sans", # serif是系统的衬线字体，sans是系统的非衬线字体; Simsun宋体,Simhei黑体
        size =12,
        face ="italic", # 常规字体(plain); 粗体(bold); 斜体(italic); 粗斜体(bold.italic) 
        margin = margin(t =5), # 横轴标题到横轴距离
        color ="black" 
      ),
      axis.title.y = element_text( # 注释同理，参考上面
        family ="sans", 
        size =12,
        face ="italic", 
        angle =90,
        margin = margin(r =5),# 纵轴标题到横轴距离
        color ="black" 
      ),
      # 3.7.3 坐标轴刻度格式设置
      axis.text.x = element_text( # 注释同理，参考3.7.2部分
        family ="serif", 
        size =10, 
        face ="bold.italic",
        angle =0, 
        margin = margin(t =3), 
        color ="black"), 
      axis.text.y = element_text(# 注释同理，参考3.7.2部分
        family ="serif",
        size =10,
        face ="bold.italic",
        angle =0,
        margin = margin(r =3),
        color ="black"),
      # 3.7.4 图形外侧边距设置（是否裁剪到图形 or 保留白边）
      plot.margin = margin(0.5,0.5,0.5,0.5,"cm") # 增加外侧边距
    ) 
 
  
  # 3.绘制纵向分组柱状图
  #---------------
    long_data$Index_group <- factor(long_data$Index_group, 
                           levels = c("Radiologists", "original report","Qwen3", "MedGemma", "ChatGPT-4.0","DeepSeek" ))
  # 3.绘制纵向分组柱状图
  #---------------
  ggplot(long_data, aes(x = 样本, y = Value, fill = Index_group)) +
    # 3.1 绘制柱状图
    geom_col(position = position_dodge(0.9), # 设置dodge宽度等于柱子宽度
             width = 0.9, # 柱子宽度
             color = "NA", # 边框颜色
             linewidth = 0.05, # 边框粗细
             alpha = 1 # 柱子透明度
    ) +
    # 3.2 给柱子填充颜色
    scale_fill_manual(values = c('#354e97',"#F2A2AF",'#c7e5ec','#f5b46f','#df5b3f','#a17db4')) +
    
    # 3.3 调整标签位置
    geom_text( 
      aes(label = Value),
      position = position_dodge(width = 0.85), # 与柱子的dodge宽度保持一致
      vjust = -0.2,  # 标签在柱子上方
      hjust = 0.5,   # 水平居中
      size = 4.5,
      angle = 0,
      family = "serif",
      fontface = "plain",
      color = "black",
      check_overlap = TRUE
    ) +
    
    # 3.4 修改标题 
    labs(x = "X_Title",  
         y = "Y_Title", 
         fill = "Index") +
    
    # 3.5 精修Y轴（数值轴）
    scale_y_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, by = 20),
      labels = function(y) sprintf("%.0f", y),
      expand = c(0, 0)
    ) +
    
    # 3.6 设置主题
    theme_classic() +
    theme(
      panel.grid.major.x = element_blank(), # 移除X轴主网格线
      panel.grid.minor.x = element_blank(), # 移除X轴次网格线
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.2), # 添加Y轴网格线
      panel.grid.minor.y = element_blank(),
      axis.ticks.length = unit(0.1, "cm"),
      legend.key.width = unit(0.6, "cm"),
      
      # 3.7.2 横纵轴标题的细节
      axis.title.x = element_text(
        family = "sans",
        size = 16,
        face = "italic",
        margin = margin(t = 5),
        color = "black" 
      ),
      axis.title.y = element_text(
        family = "sans", 
        size = 16,
        face = "italic", 
        margin = margin(r = 5),
        color = "black" 
      ),
      
      # 3.7.3 坐标轴刻度格式设置
      axis.text.x = element_text(
        family = "serif", 
        size = 16, 
        face = "bold.italic",
        angle = 0, 
        margin = margin(t = 3), 
        color = "black"), 
      axis.text.y = element_text(
        family = "serif",
        size = 16,
        face = "bold.italic", 
        margin = margin(r = 3),
        color = "black"),
      
      # 3.7.4 图形外侧边距设置
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm")
    )
  
  
   
  
  
  devtools::install_github("davidsjoberg/ggsankey")
  rm(list = ls())
  library(ggplot2)
  library(ggsankey)
  library(dplyr)
  library(patchwork)
  set.seed(123)

 
  df_T <- df %>%
    dplyr::select(T_stage, ypT_stage) %>%
    make_long(T_stage, ypT_stage) %>%
    mutate(stage = ifelse(x=="T_stage","治疗前","治疗后"))
  
  dagg <- df_T %>%
    group_by(stage, node) %>%
    summarise(n = n(), .groups ="drop") %>%
    group_by(stage) %>%
    mutate(pct =round(n / sum(n) *100,2))
  
  df2 <- df_T %>%
    left_join(dagg, by =c("stage","node"))
  
  p_T <- ggplot(
    df2,
    aes(
      x=x,
      next_x = next_x,
      node = node,
      next_node = next_node,
      fill = node,
      label = paste0(node,"\nn=", n," (", pct,"%)")
    )) +
    geom_sankey(
      flow.alpha =0.6,
      node.color ="black",
      node.width =12,
      width =0.10,
      smooth =6,
      space = 2.0,
      show.legend = FALSE
    ) +
    geom_sankey_text(
      size =3,
      color ="black",
      hjust =0.45,
      fontface ="bold"
    ) +
    scale_fill_manual(
      values=c(
        'T1'="#BBB3B1",
        'T2'="#F9F3EA",
        'T3'="#F2D3BF",
        'T4'="#F29B90",
        'T5'="#DB5376"
      )) +
    scale_x_discrete(
      labels =c("Pre-therapy","Post-therapy"),
      expand= expansion(add=0.15)
    ) +
    theme_bw() +
    theme(panel.border = element_blank(),
          axis.title = element_blank(),
          axis.text.y= element_blank(),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          axis.text.x= element_text(size =12, face ="bold"),
          plot.margin = margin(0.2, 0.2, 0.2, 0.2, "cm")  # 减少图形边距
    )
  p_T  
 
  
  library(dplyr)
  library(ggplot2)
  library(ggsankey)
  
  df <- data.frame(
    T_stage = df$T_stage,
    ypT_stage = df$ypT_stage
  )
  
  df_T <- df %>%
    dplyr::select(T_stage, ypT_stage) %>%
    make_long(T_stage, ypT_stage) %>%
    mutate(
      stage = ifelse(x == "T_stage", "治疗前", "治疗后"),
      node = factor(node, levels = c("T1", "T2", "T3", "T4", "T5"))
    )
  
  dagg <- df_T %>%
    group_by(stage, node) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(stage) %>%
    mutate(pct = round(n / sum(n) * 100, 2))
  
  df2 <- df_T %>%
    left_join(dagg, by = c("stage", "node"))
  
  p_T <- ggplot(
    df2,
    aes(
      x = x,
      next_x = next_x,
      node = node,
      next_node = next_node,
      fill = node,
      label = paste0(node, "\nn=", n, " (", pct, "%)")
    )
  ) +
    geom_sankey(
      flow.alpha = 0.6,
      node.color = "black",
      node.width = 12,
      width = 0.10,
      smooth = 6,
      space = 2.0,
      show.legend = FALSE
    ) +
    geom_sankey_text(
      size = 3,
      color = "black",
      hjust = 0.45,
      fontface = "bold"
    ) +
    scale_fill_manual(
      values = c(
        "T1" = "#BBB3B1",
        "T2" = "#F9F3EA",
        "T3" = "#F2D3BF",
        "T4" = "#F29B90",
        "T5" = "#DB5376"
      ),
      drop = FALSE
    ) +
    scale_x_discrete(
      labels = c(
        "T_stage" = "Pre-therapy",
        "ypT_stage" = "Post-therapy"
      ),
      expand = expansion(add = 0.15)
    ) +
    theme_bw() +
    theme(
      panel.border = element_blank(),
      axis.title = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      axis.text.x = element_text(size = 12, face = "bold"),
      plot.margin = margin(0.2, 0.2, 0.2, 0.2, "cm")
    )
  
  p_T
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
ggplot(tab_df, aes(x = truth, y = Qwen3, fill = Freq)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = Freq), size = 4, color = "white") +
    scale_fill_gradientn(
      colors = c("#F7FBFF", "#6BAED6", "#08306B"),  # 自定义颜色序列
      name = "病例数",
      limits = c(min(tab_df$Freq), max(tab_df$Freq)),
      oob = scales::squish
    ) +
    labs(
      title = title_main,
      x = "truth",
      y = "GPT4",
      fill = "病例数"
    ) +
    coord_fixed() +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, lineheight = 1.15, face = "bold"),
      legend.position = "right"
    )
  


library(ggplot2)
library(caret)
library(pheatmap)

# 模拟真实分类结果
set.seed(456)
n <- 200
true_labels <- factor(rep(c("Setosa", "Versicolor", "Virginica"), 
                          times = c(70, 65, 65)))
pred_labels <- true_labels

# 添加一些错误分类
pred_labels[sample(1:70, 10)] <- "Versicolor"
pred_labels[sample(71:135, 15)] <- "Virginica"
pred_labels[sample(136:200, 8)] <- "Versicolor"

# 计算混淆矩阵
real_conf_matrix <- confusionMatrix(pred_labels, true_labels)

# 转换为数据框
real_conf_df <- as.data.frame(as.table(real_conf_matrix$table))
names(real_conf_df) <- c("Predicted", "Actual", "Count")

# 计算每行的百分比（按实际类别）
real_conf_df <- real_conf_df %>%
  group_by(Actual) %>%
  mutate(Percentage = round(Count / sum(Count) * 100, 1)) %>%
  ungroup()

# 绘制精美的热力图
ggplot(real_conf_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = paste(Count, "\n(", Percentage, "%)", sep = "")), 
            color = "black", size = 4.5, fontface = "bold") +
  scale_fill_gradientn(colors = c("#f7fbff", "#6baed6", "#2171b5", "#08306b"),
                       name = "样本数量") +
  labs(title = "鸢尾花分类混淆矩阵",
       subtitle = "显示样本数量和百分比",
       x = "真实类别",
       y = "预测类别") +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(),
        axis.text = element_text(face = "bold"),
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))






# 假设 pred 是预测标签, truth 是真实标签
truth <- rating$truth
pred  <- rating$GPT4

# 计算准确率
acc <- mean(pred == truth)
n <- length(truth)
k <- sum(pred == truth)

# 用 binom.test 求置信区间
binom.test(k, n)$conf.int



# 加载包
install.packages("caret")
library(caret)
data <- dat
# 转为因子
data$truth <- factor(data$truth)
data$`original report` <- factor(data$`original report`, levels = levels(data$truth))

# 混淆矩阵
cm <- confusionMatrix(data$`original report`, data$truth)
cm
set.seed(123)

# 定义计算宏平均指标函数
get_metrics <- function(df) {
  cm <- confusionMatrix(df$`original report`, df$truth)
  byClass <- cm$byClass
  precision <- mean(byClass[, "Precision"], na.rm = TRUE)
  recall <- mean(byClass[, "Recall"], na.rm = TRUE)
  f1 <- mean(byClass[, "F1"], na.rm = TRUE)
  c(precision, recall, f1)
}

# Bootstrap 自助法
n_boot <- 2000
boot_metrics <- replicate(n_boot, {
  idx <- sample(seq_len(nrow(data)), replace = TRUE)
  get_metrics(data[idx, ])
})

# 计算均值与95%置信区间
mean_metrics <- rowMeans(boot_metrics)
ci <- apply(boot_metrics, 1, quantile, probs = c(0.025, 0.975), na.rm = TRUE)

res <- data.frame(
  Metric = c("Precision", "Recall", "F1"),
  Mean = mean_metrics,
  CI_lower = ci[1, ],
  CI_upper = ci[2, ]
)
print(res, digits = 4)







library(caret)

# 数据
data <- dat

# 模型列
models <- c("Qwen3", "Deepseek", "GPT4",  "Medgemma")
#个数
result <- data.frame(
  model = models,
  correct_n = sapply(models, function(x) sum(df[[x]] == df$truth)),
  wrong_n   = sapply(models, function(x) sum(df[[x]] != df$truth)),
  total_n   = nrow(df),
  accuracy  = sapply(models, function(x) mean(df[[x]] == df$truth))
)

result
# 因子化（统一 levels 很关键！）
data$truth <- factor(data$truth)
for (m in models) {
  data[[m]] <- factor(data[[m]], levels = levels(data$truth))
}

set.seed(123)

get_metrics <- function(df, model) {
  
  cm <- confusionMatrix(df[[model]], df$truth)
  byClass <- cm$byClass
  
  # 关键修复点👇
  if (is.null(dim(byClass))) {
    # 二分类 fallback
    precision <- byClass["Precision"]
    recall    <- byClass["Recall"]
    f1        <- byClass["F1"]
  } else {
    precision <- mean(byClass[, "Precision"], na.rm = TRUE)
    recall    <- mean(byClass[, "Recall"], na.rm = TRUE)
    f1        <- mean(byClass[, "F1"], na.rm = TRUE)
  }
  
  c(Precision = precision, Recall = recall, F1 = f1)
}

n_boot <- 2000

res_list <- lapply(models, function(model) {
  
  boot_metrics <- replicate(n_boot, {
    idx <- sample(seq_len(nrow(data)), replace = TRUE)
    get_metrics(data[idx, ], model)
  })
  
  mean_metrics <- rowMeans(boot_metrics, na.rm = TRUE)
  
  ci <- apply(
    boot_metrics,
    1,
    quantile,
    probs = c(0.025, 0.975),
    na.rm = TRUE
  )
  
  data.frame(
    Model = model,
    Metric = c("Precision", "Recall", "F1"),
    Mean = mean_metrics,
    CI_lower = ci[1, ],
    CI_upper = ci[2, ]
  )
})

res <- do.call(rbind, res_list)

print(res, digits = 4)


#########################################################################################
library(dplyr)
library(tidyr)
library(geepack)
library(lme4)
library(irr)
library(pROC)


# Read CSV
dat <- read.csv(
  "your_data.csv",
  check.names = FALSE
)

# Rename columns
dat <- dat %>%
  rename(
    qwen3 = Qwen3,
    original = `original report`,
    gpt4 = GPT4,
    junior = jr,
    medgemma = Medgemma,
    deepseek = Deepseek,
    mibc = MIBC
  )

# Variables representing VI-RADS scores
methods <- c(
  "qwen3",
  "original",
  "gpt4",
  "junior",
  "medgemma",
  "deepseek"
)

# Rename columns
dat <- dat %>%
  rename(
    duizhao = duizhao,
   
    guanjian = guanjian,
    mibc = MIBC
  )

# Variables representing VI-RADS scores
methods <- c(
  "duizhao",
  "guanjian"
)







score_vars <- c(
  "truth",
  methods
)

# Convert scores to numeric
dat[score_vars] <- lapply(
  dat[score_vars],
  as.numeric
)

dat$mibc <- as.numeric(dat$mibc)

dat$patient_id <- as.character(dat$patient_id)
dat$cohort <- as.character(dat$cohort)
dat$lesion_id <- as.character(dat$lesion_id)
# ============================================================
# Cluster bootstrap: Accuracy
# ============================================================

calc_accuracy <- function(ref, pred){
  
  ok <- complete.cases(ref, pred)
  
  mean(ref[ok] == pred[ok])
}


cluster_boot_accuracy <- function(
    data,
    pred_var,
    ref_var = "truth",
    patient_var = "patient_id",
    B = 5000,
    seed = 2026
){
  
  set.seed(seed)
  
  d <- data %>%
    filter(
      !is.na(.data[[ref_var]]),
      !is.na(.data[[pred_var]]),
      !is.na(.data[[patient_var]])
    )
  
  patient_ids <- unique(d[[patient_var]])
  n_patients <- length(patient_ids)
  
  
  # Point estimate using original data
  observed <- calc_accuracy(
    d[[ref_var]],
    d[[pred_var]]
  )
  
  
  boot_values <- rep(NA_real_, B)
  
  
  for(b in seq_len(B)){
    
    # Sample PATIENTS with replacement
    sampled_patients <- sample(
      patient_ids,
      size = n_patients,
      replace = TRUE
    )
    
    
    # Include all lesions of each sampled patient
    boot_data <- bind_rows(
      lapply(
        seq_along(sampled_patients),
        function(i){
          
          tmp <- d[
            d[[patient_var]] == sampled_patients[i],
            ,
            drop = FALSE
          ]
          
          # New cluster ID so repeated patients
          # are treated as separate bootstrap draws
          tmp$boot_id <- i
          
          tmp
        }
      )
    )
    
    
    boot_values[b] <- calc_accuracy(
      boot_data[[ref_var]],
      boot_data[[pred_var]]
    )
  }
  
  
  ci <- quantile(
    boot_values,
    c(0.025, 0.975),
    na.rm = TRUE
  )
  
  
  data.frame(
    Accuracy = observed,
    Lower95 = unname(ci[1]),
    Upper95 = unname(ci[2])
  )
}
test2 <- dat %>%
  filter(cohort == "test2")

cluster_boot_accuracy(
  data = test2,
  pred_var = "guanjian",
  B = 5000
)



# ============================================================
# Cluster bootstrap: weighted Cohen's kappa
# ============================================================

calc_kappa <- function(ref, pred){
  
  tmp <- data.frame(
    truth = ref,
    pred  = pred
  )
  
  tmp <- na.omit(tmp)
  
  # Keep VI-RADS categories consistent
  tmp$truth <- factor(
    tmp$truth,
    levels = 1:5
  )
  
  tmp$pred <- factor(
    tmp$pred,
    levels = 1:5
  )
  
  res <- irr::kappa2(
    tmp,
    weight = "weighted"
  )
  
  return(res$value)
}


cluster_boot_kappa <- function(
    data,
    pred_var,
    ref_var = "truth",
    patient_var = "patient_id",
    B = 5000,
    seed = 2026
){
  
  set.seed(seed)
  
  # Remove missing data
  d <- data %>%
    filter(
      !is.na(.data[[ref_var]]),
      !is.na(.data[[pred_var]]),
      !is.na(.data[[patient_var]])
    )
  
  
  # Unique patients
  patients <- unique(d[[patient_var]])
  n_patients <- length(patients)
  
  
  # ----------------------------------------------------------
  # Original kappa
  # ----------------------------------------------------------
  
  observed <- calc_kappa(
    d[[ref_var]],
    d[[pred_var]]
  )
  
  
  # ----------------------------------------------------------
  # Patient-level cluster bootstrap
  # ----------------------------------------------------------
  
  boot_values <- rep(
    NA_real_,
    B
  )
  
  
  for(b in seq_len(B)){
    
    # Sample patients, NOT lesions
    sampled_patients <- sample(
      patients,
      size = n_patients,
      replace = TRUE
    )
    
    
    # Include ALL lesions from each sampled patient
    boot_data <- bind_rows(
      
      lapply(
        seq_along(sampled_patients),
        function(i){
          
          tmp <- d[
            d[[patient_var]] == sampled_patients[i],
            ,
            drop = FALSE
          ]
          
          # Give repeated bootstrap draws a unique ID
          tmp$boot_id <- i
          
          tmp
        }
      )
    )
    
    
    # Calculate weighted Cohen's kappa
    boot_values[b] <- tryCatch(
      
      calc_kappa(
        boot_data[[ref_var]],
        boot_data[[pred_var]]
      ),
      
      error = function(e){
        NA_real_
      }
    )
  }
  
  
  # ----------------------------------------------------------
  # Bootstrap percentile 95% CI
  # ----------------------------------------------------------
  
  ci <- quantile(
    boot_values,
    probs = c(0.025, 0.975),
    na.rm = TRUE
  )
  
  
  data.frame(
    Kappa = observed,
    CI_lower = unname(ci[1]),
    CI_upper = unname(ci[2])
  )
}

cluster_boot_kappa(
  data = test2,
  pred_var = "duizhao",
  B = 5000
)


run_cluster_accuracy_kappa <- function(
    data,
    cohort_name,
    B = 5000
){
  
  d <- data %>%
    filter(cohort == cohort_name)
  
  
  bind_rows(
    
    lapply(
      methods,
      function(m){
        
        cat("Running:", cohort_name, m, "\n")
        
        # -------------------------
        # Cluster bootstrap Accuracy
        # -------------------------
        acc <- cluster_boot_accuracy(
          data = d,
          pred_var = m,
          ref_var = "truth",
          patient_var = "patient_id",
          B = B,
          seed = 2026
        )
        
        
        # -------------------------
        # Cluster bootstrap
        # weighted Cohen's kappa
        # -------------------------
        kap <- cluster_boot_kappa(
          data = d,
          pred_var = m,
          ref_var = "truth",
          patient_var = "patient_id",
          B = B,
          seed = 2026
        )
        
        
        # -------------------------
        # Combine results
        # -------------------------
        data.frame(
          cohort = cohort_name,
          Model = m,
          
          Accuracy = acc$Accuracy,
          Accuracy_CI_lower = acc$Lower95,
          Accuracy_CI_upper = acc$Upper95,
          
          Kappa = kap$Kappa,
          Kappa_CI_lower = kap$CI_lower,
          Kappa_CI_upper = kap$CI_upper
        )
      }
    )
  )
}
cluster_summary <- bind_rows(
  
  run_cluster_accuracy_kappa(
    data = dat,
    cohort_name = "test1",
    B = 5000
  ),
  
  run_cluster_accuracy_kappa(
    data = dat,
    cohort_name = "test2",
    B = 5000
  )
)


print(cluster_summary)



# ============================================================
# Prepare long-format data for GEE
# ============================================================

dat_long <- dat %>%
  pivot_longer(
    cols = all_of(methods),
    names_to = "method",
    values_to = "prediction"
  ) %>%
  filter(
    !is.na(truth),
    !is.na(prediction),
    !is.na(patient_id)
  ) %>%
  mutate(
    correct = as.integer(prediction == truth),
    
    # Convert patient ID to numeric cluster ID
    patient_cluster = as.integer(
      factor(patient_id)
    )
  )


# Set Qwen3 as reference
dat_long$method <- factor(dat_long$method)

dat_long$method <- relevel(
  dat_long$method,
  ref = "qwen3"
)



gee_dat_test1 <- dat_long %>%
  filter(cohort == "test1") %>%
  arrange(
    patient_cluster,
    lesion_id,
    method
  )


# Check number of clusters
length(
  unique(gee_dat_test1$patient_cluster)
)

# 应该是 308
library(geepack)

gee_test1 <- geeglm(
  correct ~ method,
  id = patient_cluster,
  data = gee_dat_test1,
  family = binomial(link = "logit"),
  corstr = "exchangeable"
)

summary(gee_test1)


gee_dat_test2 <- dat_long %>%
  filter(cohort == "test2") %>%
  arrange(
    patient_cluster,
    lesion_id,
    method
  )


length(
  unique(gee_dat_test2$patient_cluster)
)

# 应该是 103
gee_test2 <- geeglm(
  correct ~ method,
  id = patient_cluster,
  data = gee_dat_test2,
  family = binomial(link = "logit"),
  corstr = "exchangeable"
)

summary(gee_test2)

p_values <- c(0.020, 0.090, 0.001, 0.001, 0.001,0.040)

p.adjust(p_values, method = "holm")


# ============================================================
# Mixed-effects logistic regression
# ============================================================

glmm_test1 <- glmer(
  correct ~ method + (1 | patient_id),
  data = dat_long %>%
    filter(cohort == "test1"),
  family = binomial(link = "logit"),
  control = glmerControl(
    optimizer = "bobyqa"
  )
)

summary(glmm_test1)
glmm_test2 <- glmer(
  correct ~ method + (1 | patient_id),
  data = dat_long %>%
    filter(cohort == "test2"),
  family = binomial(link = "logit"),
  control = glmerControl(
    optimizer = "bobyqa"
  )
)

summary(glmm_test2)


extract_glmm <- function(model){
  
  coefs <- summary(model)$coefficients
  
  temp <- data.frame(
    term = rownames(coefs),
    estimate = coefs[, "Estimate"],
    SE = coefs[, "Std. Error"],
    z = coefs[, "z value"],
    P_value = coefs[, "Pr(>|z|)"],
    row.names = NULL
  )
  
  temp %>%
    mutate(
      OR = exp(estimate),
      OR_Lower95 = exp(
        estimate - 1.96 * SE
      ),
      OR_Upper95 = exp(
        estimate + 1.96 * SE
      )
    )
}


glmm_result_test1 <- extract_glmm(
  glmm_test1
)

glmm_result_test2 <- extract_glmm(
  glmm_test2
)

glmm_result_test1
glmm_result_test2





# ============================================================
# Cluster bootstrap:
# Macro Recall, Precision, F1 score
# ============================================================


# ------------------------------------------------------------
# 1. Calculate macro Recall / Precision / F1
# ------------------------------------------------------------

calc_macro_metrics <- function(ref, pred){
  
  tmp <- data.frame(
    truth = ref,
    pred = pred
  )
  
  tmp <- na.omit(tmp)
  
  # Keep VI-RADS categories consistent
  tmp$truth <- factor(
    tmp$truth,
    levels = 1:5
  )
  
  tmp$pred <- factor(
    tmp$pred,
    levels = 1:5
  )
  
  
  # Confusion matrix
  cm <- table(
    Truth = tmp$truth,
    Pred = tmp$pred
  )
  
  
  recall_each <- rep(NA_real_, 5)
  precision_each <- rep(NA_real_, 5)
  f1_each <- rep(NA_real_, 5)
  
  
  for(i in 1:5){
    
    TP <- cm[i, i]
    
    FN <- sum(cm[i, ]) - TP
    
    FP <- sum(cm[, i]) - TP
    
    
    # Recall
    if((TP + FN) > 0){
      recall_each[i] <- TP / (TP + FN)
    }
    
    
    # Precision
    if((TP + FP) > 0){
      precision_each[i] <- TP / (TP + FP)
    }
    
    
    # F1
    if(
      !is.na(recall_each[i]) &&
      !is.na(precision_each[i]) &&
      (recall_each[i] + precision_each[i]) > 0
    ){
      
      f1_each[i] <- 2 *
        precision_each[i] *
        recall_each[i] /
        (
          precision_each[i] +
            recall_each[i]
        )
    }
  }
  
  
  data.frame(
    
    Recall = mean(
      recall_each,
      na.rm = TRUE
    ),
    
    Precision = mean(
      precision_each,
      na.rm = TRUE
    ),
    
    F1 = mean(
      f1_each,
      na.rm = TRUE
    )
  )
}


# ============================================================
# 2. Patient-level cluster bootstrap
#    for Recall / Precision / F1
# ============================================================

cluster_boot_metrics <- function(
    data,
    pred_var,
    ref_var = "truth",
    patient_var = "patient_id",
    B = 5000,
    seed = 2026
){
  
  set.seed(seed)
  
  
  d <- data %>%
    filter(
      !is.na(.data[[ref_var]]),
      !is.na(.data[[pred_var]]),
      !is.na(.data[[patient_var]])
    )
  
  
  patients <- unique(
    d[[patient_var]]
  )
  
  n_patients <- length(
    patients
  )
  
  
  # ----------------------------------------------------------
  # Original point estimates
  # ----------------------------------------------------------
  
  observed <- calc_macro_metrics(
    d[[ref_var]],
    d[[pred_var]]
  )
  
  
  # ----------------------------------------------------------
  # Storage
  # ----------------------------------------------------------
  
  boot_recall <- rep(
    NA_real_,
    B
  )
  
  boot_precision <- rep(
    NA_real_,
    B
  )
  
  boot_f1 <- rep(
    NA_real_,
    B
  )
  
  
  # ----------------------------------------------------------
  # Patient-level cluster bootstrap
  # ----------------------------------------------------------
  
  for(b in seq_len(B)){
    
    sampled_patients <- sample(
      patients,
      size = n_patients,
      replace = TRUE
    )
    
    
    boot_data <- bind_rows(
      
      lapply(
        seq_along(sampled_patients),
        function(i){
          
          tmp <- d[
            d[[patient_var]] == sampled_patients[i],
            ,
            drop = FALSE
          ]
          
          # Unique bootstrap draw ID
          tmp$boot_id <- i
          
          tmp
        }
      )
    )
    
    
    tmp_metrics <- tryCatch(
      
      calc_macro_metrics(
        boot_data[[ref_var]],
        boot_data[[pred_var]]
      ),
      
      error = function(e){
        
        data.frame(
          Recall = NA_real_,
          Precision = NA_real_,
          F1 = NA_real_
        )
      }
    )
    
    
    boot_recall[b] <- tmp_metrics$Recall
    
    boot_precision[b] <- tmp_metrics$Precision
    
    boot_f1[b] <- tmp_metrics$F1
  }
  
  
  # ----------------------------------------------------------
  # Percentile 95% CI
  # ----------------------------------------------------------
  
  recall_ci <- quantile(
    boot_recall,
    probs = c(0.025, 0.975),
    na.rm = TRUE
  )
  
  precision_ci <- quantile(
    boot_precision,
    probs = c(0.025, 0.975),
    na.rm = TRUE
  )
  
  f1_ci <- quantile(
    boot_f1,
    probs = c(0.025, 0.975),
    na.rm = TRUE
  )
  
  
  # ----------------------------------------------------------
  # Output
  # ----------------------------------------------------------
  
  data.frame(
    
    Recall = observed$Recall,
    Recall_CI_lower = unname(
      recall_ci[1]
    ),
    Recall_CI_upper = unname(
      recall_ci[2]
    ),
    
    Precision = observed$Precision,
    Precision_CI_lower = unname(
      precision_ci[1]
    ),
    Precision_CI_upper = unname(
      precision_ci[2]
    ),
    
    F1 = observed$F1,
    F1_CI_lower = unname(
      f1_ci[1]
    ),
    F1_CI_upper = unname(
      f1_ci[2]
    )
  )
}
# ============================================================
# 3. Run all models within each cohort
# ============================================================

run_cluster_recall_precision_f1 <- function(
    data,
    cohort_name,
    B = 5000
){
  
  d <- data %>%
    filter(
      cohort == cohort_name
    )
  
  
  bind_rows(
    
    lapply(
      methods,
      function(m){
        
        cat(
          "Running:",
          cohort_name,
          m,
          "\n"
        )
        
        
        res <- cluster_boot_metrics(
          data = d,
          pred_var = m,
          ref_var = "truth",
          patient_var = "patient_id",
          B = B,
          seed = 2026
        )
        
        
        data.frame(
          
          cohort = cohort_name,
          Model = m,
          
          Recall = res$Recall,
          Recall_CI_lower = res$Recall_CI_lower,
          Recall_CI_upper = res$Recall_CI_upper,
          
          Precision = res$Precision,
          Precision_CI_lower = res$Precision_CI_lower,
          Precision_CI_upper = res$Precision_CI_upper,
          
          F1 = res$F1,
          F1_CI_lower = res$F1_CI_lower,
          F1_CI_upper = res$F1_CI_upper
        )
      }
    )
  )
}


cluster_metrics_summary <- bind_rows(
  
  run_cluster_recall_precision_f1(
    data = dat,
    cohort_name = "test1",
    B = 5000
  ),
  
  run_cluster_recall_precision_f1(
    data = dat,
    cohort_name = "test2",
    B = 5000
  )
)


print(
  cluster_metrics_summary
)





library(dplyr)

# ============================================================
# Patient-level sensitivity analysis
# Select the lesion with the highest expert-consensus VI-RADS
# ============================================================
library(dplyr)

dat <- dat %>%
  rename(
    qwen3 = Qwen3,
   
    gpt4 = GPT4,
    JR = JR,
    medgemma = Medgemma,
    deepseek = Deepseek,
    mibc = MIBC
  )
methods <- c(
  "qwen3",
  
  "gpt4",
  "JR",
  "medgemma",
  "deepseek"
)
patient_level_dat <- dat %>%
  filter(
    !is.na(patient_id),
    !is.na(truth)
  ) %>%
  group_by(
    cohort,
    patient_id
  ) %>%
  arrange(
    desc(truth),      # highest expert-consensus VI-RADS first
    lesion_id,        # tie-breaking rule
    .by_group = TRUE
  ) %>%
  slice(1) %>%
  ungroup()


# Check: each patient should now contribute only one lesion
patient_level_dat %>%
  group_by(cohort) %>%
  summarise(
    n_patients = n_distinct(patient_id),
    n_lesions = n(),
    .groups = "drop"
  )



patient_accuracy <- bind_rows(
  
  lapply(
    c("test1", "test2"),
    function(coh){
      
      d <- patient_level_dat %>%
        filter(cohort == coh)
      
      bind_rows(
        lapply(
          methods,
          function(m){
            
            data.frame(
              cohort = coh,
              Model = m,
              Accuracy = mean(
                d[[m]] == d$truth,
                na.rm = TRUE
              )
            )
          }
        )
      )
    }
  )
)

patient_accuracy


library(irr)

calc_patient_kappa <- function(ref, pred){
  
  tmp <- data.frame(
    truth = ref,
    pred = pred
  )
  
  tmp <- na.omit(tmp)
  
  tmp$truth <- factor(
    tmp$truth,
    levels = 1:5
  )
  
  tmp$pred <- factor(
    tmp$pred,
    levels = 1:5
  )
  
  irr::kappa2(
    tmp,
    weight = "weighted"
  )$value
}


patient_kappa <- bind_rows(
  
  lapply(
    c("test1", "test2"),
    function(coh){
      
      d <- patient_level_dat %>%
        filter(cohort == coh)
      
      bind_rows(
        lapply(
          methods,
          function(m){
            
            data.frame(
              cohort = coh,
              Model = m,
              Kappa = calc_patient_kappa(
                d$truth,
                d[[m]]
              )
            )
          }
        )
      )
    }
  )
)

patient_kappa


calc_macro_metrics <- function(ref, pred){
  
  tmp <- data.frame(
    truth = ref,
    pred = pred
  )
  
  tmp <- na.omit(tmp)
  
  tmp$truth <- factor(
    tmp$truth,
    levels = 1:5
  )
  
  tmp$pred <- factor(
    tmp$pred,
    levels = 1:5
  )
  
  cm <- table(
    Truth = tmp$truth,
    Pred = tmp$pred
  )
  
  recall_each <- precision_each <- f1_each <- rep(NA_real_, 5)
  
  for(i in 1:5){
    
    TP <- cm[i, i]
    FN <- sum(cm[i, ]) - TP
    FP <- sum(cm[, i]) - TP
    
    if((TP + FN) > 0){
      recall_each[i] <- TP / (TP + FN)
    }
    
    if((TP + FP) > 0){
      precision_each[i] <- TP / (TP + FP)
    }
    
    if(
      !is.na(recall_each[i]) &&
      !is.na(precision_each[i]) &&
      (recall_each[i] + precision_each[i]) > 0
    ){
      f1_each[i] <- 2 *
        recall_each[i] *
        precision_each[i] /
        (recall_each[i] + precision_each[i])
    }
  }
  
  data.frame(
    Recall = mean(recall_each, na.rm = TRUE),
    Precision = mean(precision_each, na.rm = TRUE),
    F1 = mean(f1_each, na.rm = TRUE)
  )
}
patient_metrics <- bind_rows(
  
  lapply(
    c("test1", "test2"),
    function(coh){
      
      d <- patient_level_dat %>%
        filter(cohort == coh)
      
      bind_rows(
        lapply(
          methods,
          function(m){
            
            res <- calc_macro_metrics(
              d$truth,
              d[[m]]
            )
            
            data.frame(
              cohort = coh,
              Model = m,
              Recall = res$Recall,
              Precision = res$Precision,
              F1 = res$F1
            )
          }
        )
      )
    }
  )
)

patient_metrics




# ============================================================
# Patient-level bootstrap CI
# ============================================================

patient_boot_metrics <- function(
    data,
    pred_var,
    ref_var = "truth",
    B = 5000,
    seed = 2026
){
  
  set.seed(seed)
  
  
  d <- data %>%
    filter(
      !is.na(.data[[ref_var]]),
      !is.na(.data[[pred_var]])
    )
  
  
  n <- nrow(d)
  
  
  # ----------------------------
  # Metric functions
  # ----------------------------
  
  calc_accuracy <- function(ref, pred){
    mean(ref == pred)
  }
  
  
  calc_kappa <- function(ref, pred){
    
    tmp <- data.frame(
      truth = ref,
      pred = pred
    )
    
    tmp$truth <- factor(
      tmp$truth,
      levels = 1:5
    )
    
    tmp$pred <- factor(
      tmp$pred,
      levels = 1:5
    )
    
    irr::kappa2(
      tmp,
      weight = "weighted"
    )$value
  }
  
  
  calc_macro <- function(ref, pred){
    
    tmp <- data.frame(
      truth = ref,
      pred = pred
    )
    
    tmp$truth <- factor(
      tmp$truth,
      levels = 1:5
    )
    
    tmp$pred <- factor(
      tmp$pred,
      levels = 1:5
    )
    
    
    cm <- table(
      Truth = tmp$truth,
      Pred = tmp$pred
    )
    
    
    recall <- precision <- f1 <- rep(
      NA_real_,
      5
    )
    
    
    for(i in 1:5){
      
      TP <- cm[i,i]
      
      FN <- sum(cm[i,]) - TP
      
      FP <- sum(cm[,i]) - TP
      
      
      if(TP+FN > 0)
        recall[i] <- TP/(TP+FN)
      
      if(TP+FP > 0)
        precision[i] <- TP/(TP+FP)
      
      if(
        !is.na(recall[i]) &&
        !is.na(precision[i]) &&
        recall[i]+precision[i] > 0
      ){
        f1[i] <- 2*
          recall[i]*
          precision[i]/
          (recall[i]+precision[i])
      }
    }
    
    
    c(
      Recall = mean(
        recall,
        na.rm = TRUE
      ),
      
      Precision = mean(
        precision,
        na.rm = TRUE
      ),
      
      F1 = mean(
        f1,
        na.rm = TRUE
      )
    )
  }
  
  
  # ----------------------------
  # Observed values
  # ----------------------------
  
  obs_acc <- calc_accuracy(
    d[[ref_var]],
    d[[pred_var]]
  )
  
  obs_kappa <- calc_kappa(
    d[[ref_var]],
    d[[pred_var]]
  )
  
  obs_macro <- calc_macro(
    d[[ref_var]],
    d[[pred_var]]
  )
  
  
  # ----------------------------
  # Bootstrap
  # ----------------------------
  
  boot <- matrix(
    NA,
    nrow = B,
    ncol = 5
  )
  
  colnames(boot) <- c(
    "Accuracy",
    "Kappa",
    "Recall",
    "Precision",
    "F1"
  )
  
  
  for(i in 1:B){
    
    idx <- sample(
      1:n,
      size = n,
      replace = TRUE
    )
    
    ref_b <- d[[ref_var]][idx]
    
    pred_b <- d[[pred_var]][idx]
    
    
    boot[i,"Accuracy"] <- calc_accuracy(
      ref_b,
      pred_b
    )
    
    boot[i,"Kappa"] <- tryCatch(
      calc_kappa(
        ref_b,
        pred_b
      ),
      error=function(e) NA
    )
    
    boot[i,c(
      "Recall",
      "Precision",
      "F1"
    )] <- calc_macro(
      ref_b,
      pred_b
    )
  }
  
  
  # ----------------------------
  # CI
  # ----------------------------
  
  get_ci <- function(x){
    
    q <- quantile(
      x,
      c(0.025,0.975),
      na.rm=TRUE
    )
    
    c(
      lower = q[1],
      upper = q[2]
    )
  }
  
  
  data.frame(
    
    Accuracy = obs_acc,
    Accuracy_lower = get_ci(
      boot[,"Accuracy"]
    )[1],
    Accuracy_upper = get_ci(
      boot[,"Accuracy"]
    )[2],
    
    
    Kappa = obs_kappa,
    Kappa_lower = get_ci(
      boot[,"Kappa"]
    )[1],
    Kappa_upper = get_ci(
      boot[,"Kappa"]
    )[2],
    
    
    Recall = obs_macro["Recall"],
    Recall_lower = get_ci(
      boot[,"Recall"]
    )[1],
    Recall_upper = get_ci(
      boot[,"Recall"]
    )[2],
    
    
    Precision = obs_macro["Precision"],
    Precision_lower = get_ci(
      boot[,"Precision"]
    )[1],
    Precision_upper = get_ci(
      boot[,"Precision"]
    )[2],
    
    
    F1 = obs_macro["F1"],
    F1_lower = get_ci(
      boot[,"F1"]
    )[1],
    F1_upper = get_ci(
      boot[,"F1"]
    )[2]
  )
}
patient_level_summary <- bind_rows(
  
  lapply(
    c("test1","test2"),
    function(coh){
      
      d <- patient_level_dat %>%
        filter(
          cohort == coh
        )
      
      bind_rows(
        
        lapply(
          methods,
          function(m){
            
            cat(
              "Running:",
              coh,
              m,
              "\n"
            )
            
            
            res <- patient_boot_metrics(
              data = d,
              pred_var = m,
              B = 5000
            )
            
            
            data.frame(
              
              cohort = coh,
              Model = m,
              
              res
            )
          }
        )
      )
    }
  )
)
patient_level_summary

patient_table <- patient_level_summary %>%
  mutate(
    
    Accuracy_95CI = sprintf(
      "%.3f (%.3f–%.3f)",
      Accuracy,
      Accuracy_lower,
      Accuracy_upper
    ),
    
    Kappa_95CI = sprintf(
      "%.3f (%.3f–%.3f)",
      Kappa,
      Kappa_lower,
      Kappa_upper
    ),
    
    Recall_95CI = sprintf(
      "%.3f (%.3f–%.3f)",
      Recall,
      Recall_lower,
      Recall_upper
    ),
    
    Precision_95CI = sprintf(
      "%.3f (%.3f–%.3f)",
      Precision,
      Precision_lower,
      Precision_upper
    ),
    
    F1_95CI = sprintf(
      "%.3f (%.3f–%.3f)",
      F1,
      F1_lower,
      F1_upper
    )
    
  ) %>%
  select(
    cohort,
    Model,
    Accuracy_95CI,
    Kappa_95CI,
    Recall_95CI,
    Precision_95CI,
    F1_95CI
  )


patient_table






# ============================================================
# 0. Packages
# ============================================================

library(readxl)
library(dplyr)
library(tidyr)
library(purrr)
library(irr)
library(geepack)

# ============================================================
# 1. Read and clean data
# ============================================================

dat <- read_excel("data.xlsx", sheet = "Sheet1")

dat <- dat %>%
  rename(
    Original = `original report`,
    GPT4 = GPT4,
    Radiologist = jr,
    MedGemma = Medgemma,
    DeepSeek = Deepseek
  ) %>%
  mutate(
    truth = as.numeric(truth),
    Qwen3 = as.numeric(Qwen3),
    Original = as.numeric(Original),
    GPT4 = as.numeric(GPT4),
    Radiologist = as.numeric(Radiologist),
    MedGemma = as.numeric(MedGemma),
    DeepSeek = as.numeric(DeepSeek)
  )

# Check cohort size
dat %>%
  count(cohort)

# Check observed accuracies
dat %>%
  group_by(cohort) %>%
  summarise(
    N = n(),
    Radiologist = mean(Radiologist == truth, na.rm = TRUE),
    Original = mean(Original == truth, na.rm = TRUE),
    Qwen3 = mean(Qwen3 == truth, na.rm = TRUE),
    MedGemma = mean(MedGemma == truth, na.rm = TRUE),
    GPT4 = mean(GPT4 == truth, na.rm = TRUE),
    DeepSeek = mean(DeepSeek == truth, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================
# 2. GEE function:
#    absolute paired difference in accuracy
#
# Difference = accuracy(method1) - accuracy(method2)
# ============================================================

gee_accuracy_difference <- function(data,
                                    method1,
                                    method2,
                                    cluster = "patient_id") {
  
  d <- data %>%
    filter(
      !is.na(.data[[method1]]),
      !is.na(.data[[method2]]),
      !is.na(truth),
      !is.na(.data[[cluster]])
    ) %>%
    mutate(
      # 把 P1、P2... 转成 GEE 使用的整数 cluster ID
      cluster_id = as.integer(factor(.data[[cluster]])),
      
      # 每个方法是否判断正确
      correct1 = as.numeric(.data[[method1]] == truth),
      correct2 = as.numeric(.data[[method2]] == truth),
      
      # 每个 lesion 上的配对正确率差
      diff_correct = correct1 - correct2
    ) %>%
    arrange(cluster_id)
  
  # GEE: patient-level clustering
  fit <- geeglm(
    diff_correct ~ 1,
    id = cluster_id,
    data = d,
    family = gaussian(link = "identity"),
    corstr = "exchangeable"
  )
  
  coef_tab <- summary(fit)$coefficients
  
  beta <- coef_tab["(Intercept)", "Estimate"]
  se   <- coef_tab["(Intercept)", "Std.err"]
  
  z <- beta / se
  p <- 2 * pnorm(abs(z), lower.tail = FALSE)
  
  lower <- beta - 1.96 * se
  upper <- beta + 1.96 * se
  
  tibble(
    method1 = method1,
    method2 = method2,
    
    n_lesions = nrow(d),
    n_patients = n_distinct(d[[cluster]]),
    
    accuracy_method1 = mean(d$correct1),
    accuracy_method2 = mean(d$correct2),
    
    difference = beta,
    lower95 = lower,
    upper95 = upper,
    
    standard_error = se,
    z_score = z,
    p_raw = p
  )
}
# ============================================================
# 3. Define the 6 prespecified comparisons
# ============================================================

comparisons <- tribble(
  ~method1,     ~method2,
  "Original",   "Radiologist",
  "Qwen3",      "Radiologist",
  "MedGemma",   "Radiologist",
  "GPT4",       "Radiologist",
  "DeepSeek",   "Radiologist",
  "Qwen3",      "Original"
)
# ============================================================
# 4. Run all 12 GEE comparisons
# ============================================================

all_gee_results <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map2_dfr(
      comparisons$method1,
      comparisons$method2,
      ~ gee_accuracy_difference(
        data = d_cc,
        method1 = .x,
        method2 = .y
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)

all_gee_results
all_gee_results <- all_gee_results %>%
  group_by(cohort) %>%
  mutate(
    p_holm = p.adjust(p_raw, method = "holm")
  ) %>%
  ungroup()
# ============================================================
# 6. Publication-style Table 2
# ============================================================

table2_final <- all_gee_results %>%
  mutate(
    Comparison = paste(method1, "vs", method2),
    
    Accuracy_method1 =
      sprintf("%.1f%%", 100 * accuracy_method1),
    
    Accuracy_method2 =
      sprintf("%.1f%%", 100 * accuracy_method2),
    
    Difference_pp =
      100 * difference,
    
    Lower95_pp =
      100 * lower95,
    
    Upper95_pp =
      100 * upper95,
    
    Difference_CI =
      sprintf(
        "%+.1f (%.1f to %.1f)",
        Difference_pp,
        Lower95_pp,
        Upper95_pp
      ),
    
    Raw_P =
      ifelse(
        p_raw < 0.001,
        "<0.001",
        sprintf("%.3f", p_raw)
      ),
    
    Holm_P =
      ifelse(
        p_holm < 0.001,
        "<0.001",
        sprintf("%.3f", p_holm)
      )
  ) %>%
  select(
    Comparison,
    cohort,
    Accuracy_method1,
    Accuracy_method2,
    Difference_CI,
    standard_error,
    z_score,
    Raw_P,
    Holm_P
  )
table2_final
table2_simple <- table2_final %>%
  select(
    Comparison,
    cohort,
    Accuracy_method1,
    Accuracy_method2,
    Difference_CI,
    Raw_P,
    Holm_P
  )

table2_simple

# ============================================================
# 7. Qwen3 corrected / harmed cases
# ============================================================

dat <- dat %>%
  mutate(
    original_correct = Original == truth,
    qwen_correct = Qwen3 == truth,
    
    # Original wrong -> Qwen3 correct
    qwen_corrected =
      !original_correct & qwen_correct,
    
    # Original correct -> Qwen3 wrong
    qwen_harmed =
      original_correct & !qwen_correct
  )

qwen_change_results <- dat %>%
  group_by(cohort) %>%
  summarise(
    N_lesions = n(),
    
    Qwen_corrected_n =
      sum(qwen_corrected, na.rm = TRUE),
    
    Qwen_harmed_n =
      sum(qwen_harmed, na.rm = TRUE),
    
    Patients_with_correction =
      n_distinct(patient_id[qwen_corrected]),
    
    Patients_with_harm =
      n_distinct(patient_id[qwen_harmed]),
    
    .groups = "drop"
  )

qwen_change_results



# ============================================================
# 8. weighted Cohen's kappa
# ============================================================

calc_kappa_weighted <- function(truth, prediction) {
  
  tmp <- data.frame(
    truth = truth,
    prediction = prediction
  ) %>%
    drop_na()
  
  irr::kappa2(
    tmp,
    weight = "weighted"
  )$value
}
# ============================================================
# 9. Check all weighted kappa values
# ============================================================

kappa_summary <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d <- dat %>%
      filter(cohort == cc)
    
    tibble(
      cohort = cc,
      
      Radiologist =
        calc_kappa_weighted(
          d$truth,
          d$Radiologist
        ),
      
      Original =
        calc_kappa_weighted(
          d$truth,
          d$Original
        ),
      
      Qwen3 =
        calc_kappa_weighted(
          d$truth,
          d$Qwen3
        ),
      
      MedGemma =
        calc_kappa_weighted(
          d$truth,
          d$MedGemma
        ),
      
      GPT4 =
        calc_kappa_weighted(
          d$truth,
          d$GPT4
        ),
      
      DeepSeek =
        calc_kappa_weighted(
          d$truth,
          d$DeepSeek
        )
    )
  }
)

kappa_summary
# ============================================================
# 10. Patient-cluster bootstrap:
#     formal comparison of dependent weighted kappas
# ============================================================

cluster_boot_kappa_weighted <- function(
    data,
    method1,
    method2,
    cluster = "patient_id",
    B = 10000,
    seed = 1234) {
  
  set.seed(seed)
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method1]]),
      !is.na(.data[[method2]]),
      !is.na(.data[[cluster]])
    )
  
  calc_kappa <- function(truth, pred) {
    
    tmp <- data.frame(
      truth = truth,
      pred = pred
    )
    
    irr::kappa2(
      tmp,
      weight = "weighted"
    )$value
  }
  
  # -------------------------
  # Observed kappa values
  # -------------------------
  
  k1 <- calc_kappa(
    d$truth,
    d[[method1]]
  )
  
  k2 <- calc_kappa(
    d$truth,
    d[[method2]]
  )
  
  observed_diff <- k1 - k2
  
  ids <- unique(d[[cluster]])
  
  # -------------------------
  # Cluster bootstrap
  # -------------------------
  
  boot_diff <- replicate(B, {
    
    sampled_ids <- sample(
      ids,
      size = length(ids),
      replace = TRUE
    )
    
    boot_dat <- map_dfr(
      seq_along(sampled_ids),
      function(i) {
        
        tmp <- d[
          d[[cluster]] == sampled_ids[i],
          ,
          drop = FALSE
        ]
        
        # preserve duplicated sampled clusters
        tmp$.boot_patient <- i
        
        tmp
      }
    )
    
    kb1 <- tryCatch(
      calc_kappa(
        boot_dat$truth,
        boot_dat[[method1]]
      ),
      error = function(e) NA_real_
    )
    
    kb2 <- tryCatch(
      calc_kappa(
        boot_dat$truth,
        boot_dat[[method2]]
      ),
      error = function(e) NA_real_
    )
    
    kb1 - kb2
  })
  
  boot_diff <- boot_diff[
    is.finite(boot_diff)
  ]
  
  B_valid <- length(boot_diff)
  
  # -------------------------
  # Percentile 95% CI
  # -------------------------
  
  ci <- quantile(
    boot_diff,
    probs = c(0.025, 0.975),
    na.rm = TRUE,
    names = FALSE
  )
  
  # -------------------------
  # Two-sided bootstrap P
  # Add-one correction:
  # prevents p = 0
  # -------------------------
  
  n_low  <- sum(boot_diff <= 0)
  n_high <- sum(boot_diff >= 0)
  
  p_boot <-
    2 * (min(n_low, n_high) + 1) /
    (B_valid + 1)
  
  p_boot <- min(
    p_boot,
    1
  )
  
  tibble(
    method1 = method1,
    method2 = method2,
    
    kappa_method1 = k1,
    kappa_method2 = k2,
    
    kappa_difference =
      observed_diff,
    
    lower95 =
      ci[1],
    
    upper95 =
      ci[2],
    
    p_raw =
      p_boot,
    
    B_valid =
      B_valid
  )
}

# ============================================================
# 11. All 12 formal kappa comparisons
# ============================================================

all_kappa_results <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map2_dfr(
      comparisons$method1,
      comparisons$method2,
      ~ cluster_boot_kappa_weighted(
        data = d_cc,
        method1 = .x,
        method2 = .y,
        B = 10000
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)

all_kappa_results


# ============================================================
# 12. Holm correction for formal kappa comparisons
# ============================================================

all_kappa_results <- all_kappa_results %>%
  mutate(
    p_Holm =
      p.adjust(
        p_raw,
        method = "holm"
      )
  )

all_kappa_results

# ============================================================
# 13. Publication-style formal kappa comparison table
# ============================================================

kappa_table_final <- all_kappa_results %>%
  mutate(
    Comparison =
      paste(method1, "vs", method2),
    
    Kappa_method1 =
      sprintf("%.3f", kappa_method1),
    
    Kappa_method2 =
      sprintf("%.3f", kappa_method2),
    
    Delta_kappa =
      sprintf(
        "%+.3f",
        kappa_difference
      ),
    
    Difference_CI =
      sprintf(
        "%+.3f (%.3f to %.3f)",
        kappa_difference,
        lower95,
        upper95
      ),
    
    Raw_P =
      ifelse(
        p_raw < 0.001,
        "<0.001",
        sprintf("%.3f", p_raw)
      ),
    
    Holm_P =
      ifelse(
        p_Holm < 0.001,
        "<0.001",
        sprintf("%.3f", p_Holm)
      )
  ) %>%
  select(
    Comparison,
    cohort,
    Kappa_method1,
    Kappa_method2,
    Difference_CI,
    Raw_P,
    Holm_P
  )

kappa_table_final


library(lme4)
library(marginaleffects)


lesion_glmm_accuracy <- function(
    data,
    method1,
    method2
){
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method1]]),
      !is.na(.data[[method2]]),
      !is.na(patient_id),
      !is.na(lesion_uid)
    )
  
  long <- bind_rows(
    
    d %>%
      transmute(
        patient_id,
        lesion_uid,
        method = method2,
        correct = as.integer(.data[[method2]] == truth)
      ),
    
    d %>%
      transmute(
        patient_id,
        lesion_uid,
        method = method1,
        correct = as.integer(.data[[method1]] == truth)
      )
  )
  
  long$method <- factor(
    long$method,
    levels = c(method2, method1)
  )
  
  fit <- glmer(
    correct ~ method +
      (1 | patient_id) +
      (1 | lesion_uid),
    
    data = long,
    family = binomial,
    
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 200000)
    )
  )
  
  est <- avg_comparisons(
    fit,
    variables = "method",
    type = "response",
    re.form = NA
  )
  
  tibble(
    method1 = method1,
    method2 = method2,
    
    n_patient = n_distinct(d$patient_id),
    n_lesion = nrow(d),
    
    accuracy_difference = est$estimate[1],
    lower95 = est$conf.low[1],
    upper95 = est$conf.high[1],
    p_raw = est$p.value[1],
    
    singular = isSingular(fit)
  )
}


patient_accuracy_data <- dat %>%
  group_by(
    cohort,
    patient_id
  ) %>%
  summarise(
    
    Qwen3_correct =
      mean(Qwen3 == truth),
    
    Original_correct =
      mean(Original == truth),
    
    Radiologist_correct =
      mean(Radiologist == truth),
    
    GPT4_correct =
      mean(GPT4 == truth),
    
    MedGemma_correct =
      mean(MedGemma == truth),
    
    DeepSeek_correct =
      mean(DeepSeek == truth),
    
    .groups="drop"
  )

patient_gee_accuracy <- function(
    data,
    method1,
    method2
){
  
  # 构建long format
  long <- bind_rows(
    
    data %>%
      transmute(
        patient_id,
        method = "method2",
        correct = .data[[paste0(method2, "_correct")]]
      ),
    
    data %>%
      transmute(
        patient_id,
        method = "method1",
        correct = .data[[paste0(method1, "_correct")]]
      )
    
  ) %>%
    filter(
      !is.na(patient_id),
      !is.na(correct)
    ) %>%
    mutate(
      # 避免 P1/P2 这种字符ID导致geeglm警告
      cluster_id = as.integer(factor(patient_id)),
      
      method = factor(
        method,
        levels = c("method2", "method1")
      ),
      
      correct = as.numeric(correct)
    ) %>%
    arrange(cluster_id, method)
  
  
  fit <- geeglm(
    correct ~ method,
    id = cluster_id,
    data = long,
    family = gaussian(link = "identity"),
    corstr = "exchangeable"
  )
  
  
  coef_tab <- summary(fit)$coefficients
  
  beta <- as.numeric(
    coef_tab["methodmethod1", "Estimate"]
  )
  
  se <- as.numeric(
    coef_tab["methodmethod1", "Std.err"]
  )
  
  z <- beta / se
  
  p <- 2 * pnorm(
    abs(z),
    lower.tail = FALSE
  )
  
  
  tibble(
    method1 = method1,
    method2 = method2,
    
    n_patient =
      n_distinct(long$patient_id),
    
    difference = beta,
    
    lower95 =
      beta - 1.96 * se,
    
    upper95 =
      beta + 1.96 * se,
    
    standard_error = se,
    
    z_score = z,
    
    p_raw = p
  )
}



patient_glmm_accuracy <- function(
    data,
    method1,
    method2
){
  
  long <- bind_rows(
    
    data %>%
      transmute(
        patient_id,
        method = "method2",
        correct = .data[[paste0(method2, "_correct")]]
      ),
    
    data %>%
      transmute(
        patient_id,
        method = "method1",
        correct = .data[[paste0(method1, "_correct")]]
      )
    
  ) %>%
    filter(
      !is.na(patient_id),
      !is.na(correct)
    ) %>%
    mutate(
      patient_id = factor(patient_id),
      
      method = factor(
        method,
        levels = c("method2", "method1")
      ),
      
      correct = as.numeric(correct)
    )
  
  
  fit <- glmer(
    correct ~ method +
      (1 | patient_id),
    
    data = long,
    family = binomial(link = "logit"),
    
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(
        maxfun = 200000
      )
    )
  )
  
  
  est <- avg_comparisons(
    fit,
    variables = "method",
    type = "response",
    re.form = NA
  )
  
  
  tibble(
    method1 = method1,
    method2 = method2,
    
    n_patient =
      n_distinct(long$patient_id),
    
    difference =
      as.numeric(est$estimate[1]),
    
    lower95 =
      as.numeric(est$conf.low[1]),
    
    upper95 =
      as.numeric(est$conf.high[1]),
    
    p_raw =
      as.numeric(est$p.value[1]),
    
    singular =
      isSingular(fit)
  )
}

comparisons <- tribble(
  ~method1, ~method2,
  "Original","Radiologist",
  "Qwen3","Radiologist",
  "MedGemma","Radiologist",
  "GPT4","Radiologist",
  "DeepSeek","Radiologist",
  "Qwen3","Original"
)

dat <- dat %>%
  mutate(
    lesion_uid = paste(patient_id, lesion_id, sep = "_")
  )
lesion_glmm_results <- map_dfr(
  unique(dat$cohort),
  function(cc){
    
    map2_dfr(
      comparisons$method1,
      comparisons$method2,
      
      ~lesion_glmm_accuracy(
        dat %>% filter(cohort==cc),
        .x,
        .y
      )
      
    )%>%
      mutate(cohort=cc)
    
  })

patient_gee_results <- map_dfr(
  unique(patient_accuracy_data$cohort),
  function(cc){
    
    map2_dfr(
      comparisons$method1,
      comparisons$method2,
      
      ~ patient_gee_accuracy(
        patient_accuracy_data %>%
          filter(cohort == cc),
        .x,
        .y
      )
      
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)
patient_glmm_results <- map_dfr(
  unique(patient_accuracy_data$cohort),
  function(cc){
    
    map2_dfr(
      comparisons$method1,
      comparisons$method2,
      
      ~ patient_glmm_accuracy(
        patient_accuracy_data %>%
          filter(cohort == cc),
        .x,
        .y
      )
      
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)
lesion_glmm_results$p_Holm <-
  p.adjust(
    lesion_glmm_results$p_raw,
    method="holm"
  )


patient_gee_results <- patient_gee_results %>%
  group_by(cohort) %>%
  mutate(
    p_Holm = p.adjust(
      p_raw,
      method = "holm"
    )
  ) %>%
  ungroup()


patient_gee_table_final <- patient_gee_results %>%
  mutate(
    Comparison =
      paste(method1, "vs", method2),
    
    Difference_pp =
      100 * difference,
    
    Lower95_pp =
      100 * lower95,
    
    Upper95_pp =
      100 * upper95,
    
    Difference_CI =
      sprintf(
        "%+.1f (%.1f to %.1f)",
        Difference_pp,
        Lower95_pp,
        Upper95_pp
      ),
    
    Standard_error =
      sprintf(
        "%.3f",
        standard_error
      ),
    
    Z_score =
      sprintf(
        "%.3f",
        z_score
      ),
    
    Raw_P =
      ifelse(
        p_raw < 0.001,
        "<0.001",
        sprintf("%.3f", p_raw)
      ),
    
    Holm_P =
      ifelse(
        p_Holm < 0.001,
        "<0.001",
        sprintf("%.3f", p_Holm)
      )
  ) %>%
  select(
    Comparison,
    cohort,
    n_patient,
    Difference_CI,
    Standard_error,
    Z_score,
    Raw_P,
    Holm_P
  )

patient_gee_table_final




lesion_glmm_results <- lesion_glmm_results %>%
  group_by(cohort) %>%
  mutate(
    p_Holm = p.adjust(
      p_raw,
      method = "holm"
    )
  ) %>%
  ungroup()
patient_gee_results <- patient_gee_results %>%
  group_by(cohort) %>%
  mutate(
    p_Holm = p.adjust(
      p_raw,
      method = "holm"
    )
  ) %>%
  ungroup()
lesion_glmm_table_final <- lesion_glmm_results %>%
  mutate(
    Comparison =
      paste(method1, "vs", method2),
    
    Difference_CI =
      sprintf(
        "%+.1f (%.1f to %.1f)",
        100 * accuracy_difference,
        100 * lower95,
        100 * upper95
      ),
    
    Raw_P =
      ifelse(
        p_raw < 0.001,
        "<0.001",
        sprintf("%.3f", p_raw)
      ),
    
    Holm_P =
      ifelse(
        p_Holm < 0.001,
        "<0.001",
        sprintf("%.3f", p_Holm)
      ),
    
    Singular =
      ifelse(
        singular,
        "Yes",
        "No"
      )
  ) %>%
  select(
    Comparison,
    cohort,
    n_patient,
    n_lesion,
    Difference_CI,
    Raw_P,
    Holm_P,
    Singular
  )

lesion_glmm_table_final


lesion_glmm_accuracy <- function(
    data,
    method1,
    method2
){
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method1]]),
      !is.na(.data[[method2]]),
      !is.na(patient_id),
      !is.na(lesion_uid)
    )
  
  long <- bind_rows(
    
    d %>%
      transmute(
        patient_id,
        lesion_uid,
        method = method2,
        correct = as.integer(.data[[method2]] == truth)
      ),
    
    d %>%
      transmute(
        patient_id,
        lesion_uid,
        method = method1,
        correct = as.integer(.data[[method1]] == truth)
      )
    
  ) %>%
    mutate(
      patient_id = factor(patient_id),
      lesion_uid = factor(lesion_uid),
      
      # method2作为reference
      method = factor(
        method,
        levels = c(method2, method1)
      )
    )
  
  fit <- glmer(
    correct ~ method +
      (1 | patient_id) +
      (1 | lesion_uid),
    
    data = long,
    family = binomial(link = "logit"),
    
    control = glmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 200000)
    )
  )
  
  coef_tab <- summary(fit)$coefficients
  
  # 除Intercept之外唯一的method系数
  coef_name <- grep(
    "^method",
    rownames(coef_tab),
    value = TRUE
  )[1]
  
  beta <- as.numeric(
    coef_tab[coef_name, "Estimate"]
  )
  
  se <- as.numeric(
    coef_tab[coef_name, "Std. Error"]
  )
  
  z <- as.numeric(
    coef_tab[coef_name, "z value"]
  )
  
  p <- as.numeric(
    coef_tab[coef_name, "Pr(>|z|)"]
  )
  
  tibble(
    method1 = method1,
    method2 = method2,
    
    n_patient = n_distinct(d$patient_id),
    n_lesion = nrow(d),
    
    log_OR = beta,
    standard_error = se,
    z_score = z,
    p_raw = p,
    
    OR = exp(beta),
    OR_lower95 = exp(beta - 1.96 * se),
    OR_upper95 = exp(beta + 1.96 * se),
    
    singular = isSingular(fit)
  )
}

lesion_glmm_results <- map_dfr(
  unique(dat$cohort),
  function(cc){
    
    map2_dfr(
      comparisons$method1,
      comparisons$method2,
      
      ~ lesion_glmm_accuracy(
        dat %>% filter(cohort == cc),
        .x,
        .y
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)

lesion_glmm_results <- lesion_glmm_results %>%
  group_by(cohort) %>%
  mutate(
    p_Holm = p.adjust(
      p_raw,
      method = "holm"
    )
  ) %>%
  ungroup()
lesion_glmm_results %>%
  select(
    cohort,
    method1,
    method2,
    standard_error,
    z_score,
    p_raw,
    OR,
    OR_lower95,
    OR_upper95,
    singular
  )
lesion_glmm_table_final <- lesion_glmm_results %>%
  mutate(
    Comparison =
      paste(method1, "vs", method2),
    
    OR_CI =
      sprintf(
        "%.2f (%.2f to %.2f)",
        OR,
        OR_lower95,
        OR_upper95
      ),
    
    Raw_P =
      ifelse(
        p_raw < 0.001,
        "<0.001",
        sprintf("%.3f", p_raw)
      ),
    
    Holm_P =
      ifelse(
        p_Holm < 0.001,
        "<0.001",
        sprintf("%.3f", p_Holm)
      )
  ) %>%
  select(
    Comparison,
    cohort,
    n_patient,
    n_lesion,
    OR_CI,
    Raw_P,
    Holm_P,
    singular
  )

lesion_glmm_table_final

# ============================================================
# Patient-level categorical score
# Use only if highest-risk lesion is clinically justified
# ============================================================

safe_max <- function(x) {
  
  if (all(is.na(x))) {
    return(NA_real_)
  }
  
  max(x, na.rm = TRUE)
}


patient_score_dat <- dat %>%
  group_by(
    cohort,
    patient_id
  ) %>%
  summarise(
    n_lesions = n(),
    
    truth =
      safe_max(truth),
    
    Radiologist =
      safe_max(Radiologist),
    
    Original =
      safe_max(Original),
    
    Qwen3 =
      safe_max(Qwen3),
    
    MedGemma =
      safe_max(MedGemma),
    
    GPT4 =
      safe_max(GPT4),
    
    DeepSeek =
      safe_max(DeepSeek),
    
    .groups = "drop"
  )

# ============================================================
# Patient-level weighted Cohen's kappa
# Δkappa + 95% CI
# Patient bootstrap
# ============================================================

patient_kappa_difference <- function(
    data,
    method1,
    method2,
    B = 10000,
    seed = 1234) {
  
  set.seed(seed)
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method1]]),
      !is.na(.data[[method2]]),
      !is.na(patient_id)
    )
  
  # ----------------------------------------------------------
  # Observed kappas
  # ----------------------------------------------------------
  
  k1 <- calc_weighted_kappa(
    d$truth,
    d[[method1]]
  )
  
  k2 <- calc_weighted_kappa(
    d$truth,
    d[[method2]]
  )
  
  delta_obs <- k1 - k2
  
  n <- nrow(d)
  
  # ----------------------------------------------------------
  # Patient bootstrap
  # ----------------------------------------------------------
  
  boot_delta <- replicate(
    B,
    {
      
      idx <- sample(
        seq_len(n),
        size = n,
        replace = TRUE
      )
      
      boot_dat <- d[idx, , drop = FALSE]
      
      kb1 <- calc_weighted_kappa(
        boot_dat$truth,
        boot_dat[[method1]]
      )
      
      kb2 <- calc_weighted_kappa(
        boot_dat$truth,
        boot_dat[[method2]]
      )
      
      kb1 - kb2
    }
  )
  
  boot_delta <- boot_delta[
    is.finite(boot_delta)
  ]
  
  B_valid <- length(boot_delta)
  
  # ----------------------------------------------------------
  # 95% CI
  # ----------------------------------------------------------
  
  ci <- quantile(
    boot_delta,
    probs = c(0.025, 0.975),
    na.rm = TRUE,
    names = FALSE
  )
  
  # ----------------------------------------------------------
  # Two-sided bootstrap P
  # ----------------------------------------------------------
  
  n_low  <- sum(boot_delta <= 0)
  n_high <- sum(boot_delta >= 0)
  
  p_boot <- 2 *
    (min(n_low, n_high) + 1) /
    (B_valid + 1)
  
  p_boot <- min(p_boot, 1)
  
  tibble(
    method1 = method1,
    method2 = method2,
    
    n_patients = n,
    
    kappa_method1 = k1,
    kappa_method2 = k2,
    
    delta_kappa = delta_obs,
    
    lower95 = ci[1],
    upper95 = ci[2],
    
    p_raw = p_boot,
    
    B_valid = B_valid
  )
}

patient_kappa_results <- map_dfr(
  unique(patient_score_dat$cohort),
  function(cc) {
    
    d_cc <- patient_score_dat %>%
      filter(cohort == cc)
    
    map2_dfr(
      comparisons$method1,
      comparisons$method2,
      ~ patient_kappa_difference(
        data = d_cc,
        method1 = .x,
        method2 = .y,
        B = 10000
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)
patient_kappa_results <- patient_kappa_results %>%
  mutate(
    p_Holm = p.adjust(
      p_raw,
      method = "holm"
    )
  )

patient_kappa_results


patient_kappa_table <- patient_kappa_results %>%
  mutate(
    Comparison =
      paste(method1, "vs", method2),
    
    Kappa_method1 =
      sprintf("%.3f", kappa_method1),
    
    Kappa_method2 =
      sprintf("%.3f", kappa_method2),
    
    DeltaKappa_CI =
      sprintf(
        "%+.3f (%.3f to %.3f)",
        delta_kappa,
        lower95,
        upper95
      ),
    
    Raw_P =
      ifelse(
        p_raw < 0.001,
        "<0.001",
        sprintf("%.3f", p_raw)
      ),
    
    Holm_P =
      ifelse(
        p_Holm < 0.001,
        "<0.001",
        sprintf("%.3f", p_Holm)
      )
  ) %>%
  select(
    Comparison,
    cohort,
    n_patients,
    Kappa_method1,
    Kappa_method2,
    DeltaKappa_CI,
    Raw_P,
    Holm_P
  )

patient_kappa_table



# ============================================================
# Overall ordinal error metrics
# Exact accuracy
# Accuracy within ±1 category
# Major errors >= 2 categories
# Under-classification
# Over-classification
# ============================================================

ordinal_metrics <- function(data, method) {
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method]])
    )
  
  pred <- d[[method]]
  ref  <- d$truth
  
  tibble(
    method = method,
    n = length(ref),
    
    exact_accuracy =
      mean(pred == ref),
    
    within_1_accuracy =
      mean(abs(pred - ref) <= 1),
    
    major_error_rate =
      mean(abs(pred - ref) >= 2),
    
    under_classification_rate =
      mean(pred < ref),
    
    over_classification_rate =
      mean(pred > ref)
  )
}
ordinal_results <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map_dfr(
      methods,
      ~ ordinal_metrics(
        data = d_cc,
        method = .x
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)
ordinal_results <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map_dfr(
      methods,
      ~ ordinal_metrics(
        data = d_cc,
        method = .x
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)

ordinal_results
ordinal_table <- ordinal_results %>%
  mutate(
    Exact_accuracy =
      sprintf("%.1f%%", 100 * exact_accuracy),
    
    Within_1_accuracy =
      sprintf("%.1f%%", 100 * within_1_accuracy),
    
    Major_error =
      sprintf("%.1f%%", 100 * major_error_rate),
    
    Under_classification =
      sprintf("%.1f%%", 100 * under_classification_rate),
    
    Over_classification =
      sprintf("%.1f%%", 100 * over_classification_rate)
  ) %>%
  select(
    cohort,
    method,
    n,
    Exact_accuracy,
    Within_1_accuracy,
    Major_error,
    Under_classification,
    Over_classification
  )

ordinal_table


# ============================================================
# Class-specific sensitivity and precision
# ============================================================

class_metrics <- function(data, method) {
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method]])
    )
  
  pred <- d[[method]]
  ref  <- d$truth
  
  map_dfr(
    1:5,
    function(cls) {
      
      TP <- sum(
        pred == cls &
          ref == cls
      )
      
      FN <- sum(
        pred != cls &
          ref == cls
      )
      
      FP <- sum(
        pred == cls &
          ref != cls
      )
      
      TN <- sum(
        pred != cls &
          ref != cls
      )
      
      sensitivity <-
        ifelse(
          TP + FN == 0,
          NA_real_,
          TP / (TP + FN)
        )
      
      precision <-
        ifelse(
          TP + FP == 0,
          NA_real_,
          TP / (TP + FP)
        )
      
      tibble(
        method = method,
        VI_RADS = cls,
        n_reference = TP + FN,
        n_predicted = TP + FP,
        sensitivity = sensitivity,
        precision = precision
      )
    }
  )
}

class_results <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map_dfr(
      methods,
      ~ class_metrics(
        data = d_cc,
        method = .x
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)

class_results

class_table <- class_results %>%
  mutate(
    Sensitivity =
      sprintf(
        "%.1f%%",
        100 * sensitivity
      ),
    
    Precision =
      sprintf(
        "%.1f%%",
        100 * precision
      )
  ) %>%
  select(
    cohort,
    method,
    VI_RADS,
    n_reference,
    n_predicted,
    Sensitivity,
    Precision
  )

class_table

# ============================================================
# VI-RADS 3 specific performance
# ============================================================

virads3_metrics <- function(data, method) {
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method]])
    )
  
  pred <- d[[method]]
  ref  <- d$truth
  
  TP <- sum(
    pred == 3 &
      ref == 3
  )
  
  FN <- sum(
    pred != 3 &
      ref == 3
  )
  
  FP <- sum(
    pred == 3 &
      ref != 3
  )
  
  sensitivity_3 <-
    ifelse(
      TP + FN == 0,
      NA_real_,
      TP / (TP + FN)
    )
  
  precision_3 <-
    ifelse(
      TP + FP == 0,
      NA_real_,
      TP / (TP + FP)
    )
  
  true3 <- d %>%
    filter(truth == 3)
  
  tibble(
    method = method,
    
    n_true_VIRADS3 =
      nrow(true3),
    
    sensitivity_VIRADS3 =
      sensitivity_3,
    
    precision_VIRADS3 =
      precision_3,
    
    predicted_1 =
      sum(true3[[method]] == 1),
    
    predicted_2 =
      sum(true3[[method]] == 2),
    
    predicted_3 =
      sum(true3[[method]] == 3),
    
    predicted_4 =
      sum(true3[[method]] == 4),
    
    predicted_5 =
      sum(true3[[method]] == 5),
    
    underclassification_true3 =
      mean(true3[[method]] < 3),
    
    overclassification_true3 =
      mean(true3[[method]] > 3)
  )
}

virads3_results <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map_dfr(
      methods,
      ~ virads3_metrics(
        data = d_cc,
        method = .x
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)

virads3_results

virads3_table <- virads3_results %>%
  mutate(
    Sensitivity =
      sprintf(
        "%.1f%%",
        100 * sensitivity_VIRADS3
      ),
    
    Precision =
      sprintf(
        "%.1f%%",
        100 * precision_VIRADS3
      ),
    
    Underclassification =
      sprintf(
        "%.1f%%",
        100 * underclassification_true3
      ),
    
    Overclassification =
      sprintf(
        "%.1f%%",
        100 * overclassification_true3
      )
  ) %>%
  select(
    cohort,
    method,
    n_true_VIRADS3,
    Sensitivity,
    Precision,
    predicted_1,
    predicted_2,
    predicted_3,
    predicted_4,
    predicted_5,
    Underclassification,
    Overclassification
  )

virads3_table

# ============================================================
# Binary classification:
# VI-RADS 1–3 vs VI-RADS 4–5
# Positive = VI-RADS 4–5
# ============================================================

binary_13_vs_45 <- function(data, method) {
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method]])
    )
  
  truth_binary <-
    ifelse(
      d$truth >= 4,
      1,
      0
    )
  
  pred_binary <-
    ifelse(
      d[[method]] >= 4,
      1,
      0
    )
  
  TP <- sum(
    truth_binary == 1 &
      pred_binary == 1
  )
  
  TN <- sum(
    truth_binary == 0 &
      pred_binary == 0
  )
  
  FP <- sum(
    truth_binary == 0 &
      pred_binary == 1
  )
  
  FN <- sum(
    truth_binary == 1 &
      pred_binary == 0
  )
  
  sensitivity <-
    TP / (TP + FN)
  
  specificity <-
    TN / (TN + FP)
  
  precision <-
    TP / (TP + FP)
  
  npv <-
    TN / (TN + FN)
  
  accuracy <-
    (TP + TN) /
    (TP + TN + FP + FN)
  
  f1 <-
    2 * TP /
    (2 * TP + FP + FN)
  
  tibble(
    method = method,
    
    n = nrow(d),
    
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN,
    
    sensitivity =
      sensitivity,
    
    specificity =
      specificity,
    
    precision =
      precision,
    
    NPV =
      npv,
    
    accuracy =
      accuracy,
    
    F1 =
      f1
  )
}

binary_results <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map_dfr(
      methods,
      ~ binary_13_vs_45(
        data = d_cc,
        method = .x
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)

binary_results
binary_table <- binary_results %>%
  mutate(
    Sensitivity =
      sprintf("%.1f%%", 100 * sensitivity),
    
    Specificity =
      sprintf("%.1f%%", 100 * specificity),
    
    Precision =
      sprintf("%.1f%%", 100 * precision),
    
    NPV =
      sprintf("%.1f%%", 100 * NPV),
    
    Accuracy =
      sprintf("%.1f%%", 100 * accuracy),
    
    F1 =
      sprintf("%.3f", F1)
  ) %>%
  select(
    cohort,
    method,
    n,
    TP,
    TN,
    FP,
    FN,
    Sensitivity,
    Specificity,
    Precision,
    NPV,
    Accuracy,
    F1
  )

binary_table

library(dplyr)
library(purrr)
library(tibble)

# ============================================================
# Wilson 95% CI
# ============================================================

wilson_ci <- function(x, n, conf.level = 0.95) {
  
  if (is.na(n) || n == 0) {
    return(c(NA_real_, NA_real_))
  }
  
  z <- qnorm(1 - (1 - conf.level) / 2)
  
  p <- x / n
  
  denom <- 1 + z^2 / n
  
  center <-
    (p + z^2 / (2 * n)) / denom
  
  half <-
    z *
    sqrt(
      p * (1 - p) / n +
        z^2 / (4 * n^2)
    ) / denom
  
  c(
    lower = center - half,
    upper = center + half
  )
}
# ============================================================
# VI-RADS 1–3 vs 4–5
# Positive = VI-RADS 4–5
# Wilson 95% CI for all major performance metrics
# ============================================================

binary_13_vs_45 <- function(data, method) {
  
  d <- data %>%
    filter(
      !is.na(truth),
      !is.na(.data[[method]])
    )
  
  truth_binary <- ifelse(
    d$truth >= 4,
    1,
    0
  )
  
  pred_binary <- ifelse(
    d[[method]] >= 4,
    1,
    0
  )
  
  TP <- sum(
    truth_binary == 1 &
      pred_binary == 1
  )
  
  TN <- sum(
    truth_binary == 0 &
      pred_binary == 0
  )
  
  FP <- sum(
    truth_binary == 0 &
      pred_binary == 1
  )
  
  FN <- sum(
    truth_binary == 1 &
      pred_binary == 0
  )
  
  N <- TP + TN + FP + FN
  
  # ----------------------------------------------------------
  # Point estimates
  # ----------------------------------------------------------
  
  sensitivity <- ifelse(
    TP + FN == 0,
    NA_real_,
    TP / (TP + FN)
  )
  
  specificity <- ifelse(
    TN + FP == 0,
    NA_real_,
    TN / (TN + FP)
  )
  
  ppv <- ifelse(
    TP + FP == 0,
    NA_real_,
    TP / (TP + FP)
  )
  
  npv <- ifelse(
    TN + FN == 0,
    NA_real_,
    TN / (TN + FN)
  )
  
  accuracy <- ifelse(
    N == 0,
    NA_real_,
    (TP + TN) / N
  )
  
  # ----------------------------------------------------------
  # Wilson 95% CI
  # ----------------------------------------------------------
  
  sens_ci <- wilson_ci(
    TP,
    TP + FN
  )
  
  spec_ci <- wilson_ci(
    TN,
    TN + FP
  )
  
  ppv_ci <- wilson_ci(
    TP,
    TP + FP
  )
  
  npv_ci <- wilson_ci(
    TN,
    TN + FN
  )
  
  acc_ci <- wilson_ci(
    TP + TN,
    N
  )
  
  # ----------------------------------------------------------
  # Return
  # ----------------------------------------------------------
  
  tibble(
    method = method,
    n = N,
    
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN,
    
    sensitivity = sensitivity,
    sensitivity_lower95 = sens_ci[1],
    sensitivity_upper95 = sens_ci[2],
    
    specificity = specificity,
    specificity_lower95 = spec_ci[1],
    specificity_upper95 = spec_ci[2],
    
    PPV = ppv,
    PPV_lower95 = ppv_ci[1],
    PPV_upper95 = ppv_ci[2],
    
    NPV = npv,
    NPV_lower95 = npv_ci[1],
    NPV_upper95 = npv_ci[2],
    
    accuracy = accuracy,
    accuracy_lower95 = acc_ci[1],
    accuracy_upper95 = acc_ci[2]
  )
}
methods <- c(
  "Radiologist",
  "Original",
  "Qwen3",
  "DeepSeek",
  "GPT4",
  "MedGemma"
)

binary_results <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map_dfr(
      methods,
      ~ binary_13_vs_45(
        data = d_cc,
        method = .x
      )
    ) %>%
      mutate(
        cohort = cc,
        .before = 1
      )
  }
)

binary_results
binary_table <- binary_results %>%
  mutate(
    
    Sensitivity =
      sprintf(
        "%.1f%% (%.1f–%.1f)",
        100 * sensitivity,
        100 * sensitivity_lower95,
        100 * sensitivity_upper95
      ),
    
    Specificity =
      sprintf(
        "%.1f%% (%.1f–%.1f)",
        100 * specificity,
        100 * specificity_lower95,
        100 * specificity_upper95
      ),
    
    PPV =
      sprintf(
        "%.1f%% (%.1f–%.1f)",
        100 * PPV,
        100 * PPV_lower95,
        100 * PPV_upper95
      ),
    
    NPV =
      sprintf(
        "%.1f%% (%.1f–%.1f)",
        100 * NPV,
        100 * NPV_lower95,
        100 * NPV_upper95
      ),
    
    Accuracy =
      sprintf(
        "%.1f%% (%.1f–%.1f)",
        100 * accuracy,
        100 * accuracy_lower95,
        100 * accuracy_upper95
      )
  ) %>%
  select(
    cohort,
    method,
    n,
    TP,
    TN,
    FP,
    FN,
    Sensitivity,
    Specificity,
    PPV,
    NPV,
    Accuracy
  )

binary_table

methods <- c(
  "Radiologist",
  "Original",
  "Qwen3",
  "DeepSeek",
  "GPT4",
  "MedGemma"
)
library(dplyr)
library(purrr)

error_summary <- map_dfr(
  methods,
  function(m){
    
    dat %>%
      group_by(cohort) %>%
      summarise(
        Model = m,
        Total = n(),
        Correct = sum(.data[[m]] == truth, na.rm = TRUE),
        Errors = sum(.data[[m]] != truth, na.rm = TRUE),
        Error_rate = Errors / Total,
        .groups = "drop"
      )
  }
)

error_summary

library(dplyr)
library(purrr)
library(pROC)
library(tibble)
cluster_boot_auc_difference <- function(
    data,
    outcome,
    method1,
    method2,
    patient_id = "patient_id",
    B = 10000,
    seed = 1234) {
  
  set.seed(seed)
  
  d <- data %>%
    filter(
      !is.na(.data[[outcome]]),
      !is.na(.data[[method1]]),
      !is.na(.data[[method2]]),
      !is.na(.data[[patient_id]])
    )
  
  # ----------------------------------------------------------
  # Observed AUCs
  # ----------------------------------------------------------
  
  auc1 <- as.numeric(
    pROC::roc(
      response = d[[outcome]],
      predictor = d[[method1]],
      quiet = TRUE,
      direction = "<"
    )$auc
  )
  
  auc2 <- as.numeric(
    pROC::roc(
      response = d[[outcome]],
      predictor = d[[method2]],
      quiet = TRUE,
      direction = "<"
    )$auc
  )
  
  delta_obs <- auc1 - auc2
  
  # ----------------------------------------------------------
  # Patient-level cluster bootstrap
  # ----------------------------------------------------------
  
  patients <- unique(d[[patient_id]])
  n_patients <- length(patients)
  
  boot_delta <- replicate(
    B,
    {
      
      sampled_patients <- sample(
        patients,
        size = n_patients,
        replace = TRUE
      )
      
      # Important:
      # repeated sampled patients must remain repeated clusters
      boot_dat <- map2_dfr(
        sampled_patients,
        seq_along(sampled_patients),
        function(pid, cluster_index) {
          
          d %>%
            filter(.data[[patient_id]] == pid) %>%
            mutate(
              boot_cluster = cluster_index
            )
        }
      )
      
      # Need both outcome classes
      if (length(unique(boot_dat[[outcome]])) < 2) {
        return(NA_real_)
      }
      
      a1 <- tryCatch(
        as.numeric(
          pROC::roc(
            response = boot_dat[[outcome]],
            predictor = boot_dat[[method1]],
            quiet = TRUE,
            direction = "<"
          )$auc
        ),
        error = function(e) NA_real_
      )
      
      a2 <- tryCatch(
        as.numeric(
          pROC::roc(
            response = boot_dat[[outcome]],
            predictor = boot_dat[[method2]],
            quiet = TRUE,
            direction = "<"
          )$auc
        ),
        error = function(e) NA_real_
      )
      
      a1 - a2
    }
  )
  
  boot_delta <- boot_delta[
    is.finite(boot_delta)
  ]
  
  B_valid <- length(boot_delta)
  
  # ----------------------------------------------------------
  # Percentile 95% CI
  # ----------------------------------------------------------
  
  ci <- quantile(
    boot_delta,
    probs = c(0.025, 0.975),
    names = FALSE
  )
  
  # ----------------------------------------------------------
  # Two-sided bootstrap P value
  # H0: delta AUC = 0
  # ----------------------------------------------------------
  
  n_low  <- sum(boot_delta <= 0)
  n_high <- sum(boot_delta >= 0)
  
  p_boot <- 2 *
    (min(n_low, n_high) + 1) /
    (B_valid + 1)
  
  p_boot <- min(p_boot, 1)
  
  tibble(
    method1 = method1,
    method2 = method2,
    
    lesions = nrow(d),
    patients = n_patients,
    
    auc_method1 = auc1,
    auc_method2 = auc2,
    
    delta_auc = delta_obs,
    
    lower95 = ci[1],
    upper95 = ci[2],
    
    p_value = p_boot,
    
    B_valid = B_valid
  )
}


auc_diff_lesion <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    cluster_boot_auc_difference(
      data = d_cc,
      outcome = "MIBC",
      method1 = "Qwen3",
      method2 = "Original",
      patient_id = "patient_id",
      B = 10000,
      seed = 1234
    ) %>%
      mutate(
        cohort = cc,
        analysis_level = "Lesion",
        .before = 1
      )
  }
)

auc_diff_lesion



method_map <- c(
  "Reference"        = "truth",
  "Radiologists"     = "Radiologist",
  "Original Reports" = "Original",
  "Qwen3"            = "Qwen3",
  "MedGemma"         = "MedGemma",
  "GPT-4.0"          = "GPT4",
  "DeepSeek"         = "DeepSeek"
)

method_pairs <- combn(
  names(method_map),
  2,
  simplify = FALSE
)

length(method_pairs)
# 21



auc_pairwise_lesion <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    map_dfr(
      method_pairs,
      function(pair) {
        
        label1 <- pair[1]
        label2 <- pair[2]
        
        var1 <- unname(method_map[label1])
        var2 <- unname(method_map[label2])
        
        cluster_boot_auc_difference(
          data = d_cc,
          outcome = "MIBC",
          method1 = var1,
          method2 = var2,
          patient_id = "patient_id",
          B = 10000,
          seed = 1234
        ) %>%
          mutate(
            method1_label = label1,
            method2_label = label2
          )
      }
    ) %>%
      mutate(
        cohort = cc,
        analysis_level = "Lesion",
        .before = 1
      )
  }
)
au0c_pairwise_lesion <- auc_pairwise_lesion %>%
  group_by(cohort) %>%
  mutate(
    p_holm = p.adjust(
      p_value,
      method = "holm"
    )
  ) %>%
  ungroup()
auc_pairwise_table <- auc_pairwise_lesion %>%
  mutate(
    
    Comparison = paste(
      method1_label,
      "vs",
      method2_label
    ),
    
    `AUC 1` = sprintf(
      "%.3f",
      auc_method1
    ),
    
    `AUC 2` = sprintf(
      "%.3f",
      auc_method2
    ),
    
    `ΔAUC (95% CI)` = sprintf(
      "%.3f (%.3f to %.3f)",
      delta_auc,
      lower95,
      upper95
    ),
    
    `P value` = case_when(
      p_value < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", p_value)
    ),
    
    `Holm-adjusted P` = case_when(
      p_holm < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", p_holm)
    )
  ) %>%
  select(
    cohort,
    Comparison,
    `AUC 1`,
    `AUC 2`,
    `ΔAUC (95% CI)`,
    `P value`,
    `Holm-adjusted P`
  )

auc_pairwise_table
auc_pairwise_lesion <- auc_pairwise_lesion %>%
  group_by(cohort) %>%
  mutate(
    p_holm = p.adjust(
      p_value,
      method = "holm"
    )
  ) %>%
  ungroup()

auc_pairwise_table <- auc_pairwise_lesion %>%
  mutate(
    
    Comparison = paste(
      method1_label,
      "vs",
      method2_label
    ),
    
    `AUC 1` = sprintf(
      "%.3f",
      auc_method1
    ),
    
    `AUC 2` = sprintf(
      "%.3f",
      auc_method2
    ),
    
    `ΔAUC (95% CI)` = sprintf(
      "%.3f (%.3f to %.3f)",
      delta_auc,
      lower95,
      upper95
    ),
    
    `P value` = case_when(
      p_value < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", p_value)
    ),
    
    `Holm-adjusted P` = case_when(
      p_holm < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", p_holm)
    )
  ) %>%
  select(
    cohort,
    Comparison,
    `AUC 1`,
    `AUC 2`,
    `ΔAUC (95% CI)`,
    `P value`,
    `Holm-adjusted P`
  )

auc_pairwise_table
qwen_original_final <- auc_pairwise_lesion %>%
  filter(
    (method1_label == "Qwen3" &
       method2_label == "Original Reports") |
      (method1_label == "Original Reports" &
         method2_label == "Qwen3")
  ) %>%
  mutate(
    
    delta_qwen_original = ifelse(
      method1_label == "Qwen3",
      delta_auc,
      -delta_auc
    ),
    
    lower_qwen_original = ifelse(
      method1_label == "Qwen3",
      lower95,
      -upper95
    ),
    
    upper_qwen_original = ifelse(
      method1_label == "Qwen3",
      upper95,
      -lower95
    ),
    
    result = sprintf(
      "%.3f (95%% CI %.3f to %.3f), P = %.3f",
      delta_qwen_original,
      lower_qwen_original,
      upper_qwen_original,
      p_value
    )
  ) %>%
  select(
    cohort,
    delta_qwen_original,
    lower_qwen_original,
    upper_qwen_original,
    p_value,
    p_holm,
    result
  )

qwen_original_final



library(dplyr)
library(purrr)
library(tidyr)
library(pROC)
library(tibble)

# ============================================================
# 1. Fast cluster bootstrap for ALL pairwise AUC comparisons
# ============================================================

cluster_boot_all_auc <- function(
    data,
    outcome,
    method_map,
    patient_id,
    B = 10000,
    seed = 1234
) {
  
  set.seed(seed)
  
  method_labels <- names(method_map)
  method_vars   <- unname(method_map)
  
  # check required columns
  required_cols <- c(outcome, patient_id, method_vars)
  missing_cols <- setdiff(required_cols, names(data))
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  # ----------------------------------------------------------
  # Function to calculate AUC safely
  # ----------------------------------------------------------
  
  safe_auc <- function(y, x) {
    
    ok <- !is.na(y) & !is.na(x)
    
    y <- y[ok]
    x <- x[ok]
    
    # Need both outcome classes
    if (length(unique(y)) < 2) {
      return(NA_real_)
    }
    
    tryCatch(
      as.numeric(
        pROC::auc(
          response = y,
          predictor = x,
          quiet = TRUE,
          direction = "<"
        )
      ),
      error = function(e) NA_real_
    )
  }
  
  # ----------------------------------------------------------
  # Observed AUCs
  # ----------------------------------------------------------
  
  observed_auc <- map_dbl(
    method_vars,
    ~ safe_auc(
      data[[outcome]],
      data[[.x]]
    )
  )
  
  names(observed_auc) <- method_labels
  
  # all method pairs
  pair_mat <- combn(method_labels, 2)
  
  pair_df <- tibble(
    method1 = pair_mat[1, ],
    method2 = pair_mat[2, ]
  )
  
  pair_df <- pair_df %>%
    mutate(
      auc1 = observed_auc[method1],
      auc2 = observed_auc[method2],
      difference = auc1 - auc2
    )
  
  # ----------------------------------------------------------
  # Cluster bootstrap
  # ----------------------------------------------------------
  
  patient_ids <- unique(data[[patient_id]])
  n_patients <- length(patient_ids)
  
  # matrix: B × number of methods
  boot_auc <- matrix(
    NA_real_,
    nrow = B,
    ncol = length(method_labels),
    dimnames = list(NULL, method_labels)
  )
  
  for (b in seq_len(B)) {
    
    sampled_ids <- sample(
      patient_ids,
      size = n_patients,
      replace = TRUE
    )
    
    # Important:
    # If a patient is sampled more than once, duplicate all lesions
    # belonging to that patient.
    boot_data <- map_dfr(
      seq_along(sampled_ids),
      function(i) {
        
        id <- sampled_ids[i]
        
        data %>%
          filter(.data[[patient_id]] == id) %>%
          mutate(.boot_cluster = i)
      }
    )
    
    # calculate all method AUCs on SAME bootstrap sample
    boot_auc[b, ] <- map_dbl(
      method_vars,
      ~ safe_auc(
        boot_data[[outcome]],
        boot_data[[.x]]
      )
    )
    
    # optional progress
    if (b %% 500 == 0) {
      message("Bootstrap: ", b, "/", B)
    }
  }
  
  # ----------------------------------------------------------
  # Calculate all pairwise bootstrap differences
  # ----------------------------------------------------------
  
  results <- map_dfr(
    seq_len(nrow(pair_df)),
    function(i) {
      
      m1 <- pair_df$method1[i]
      m2 <- pair_df$method2[i]
      
      obs_diff <- pair_df$difference[i]
      
      boot_diff <- boot_auc[, m1] - boot_auc[, m2]
      boot_diff <- boot_diff[!is.na(boot_diff)]
      
      # percentile bootstrap CI
      ci <- quantile(
        boot_diff,
        probs = c(0.025, 0.975),
        na.rm = TRUE,
        names = FALSE
      )
      
      # two-sided bootstrap P value
      # proportion on either side of zero
      p_lower <- mean(boot_diff <= 0)
      p_upper <- mean(boot_diff >= 0)
      
      p_value <- min(
        1,
        2 * min(p_lower, p_upper)
      )
      
      tibble(
        method1 = m1,
        method2 = m2,
        auc1 = pair_df$auc1[i],
        auc2 = pair_df$auc2[i],
        difference = obs_diff,
        ci_lower = ci[1],
        ci_upper = ci[2],
        p_value = p_value,
        n_boot_valid = length(boot_diff)
      )
    }
  )
  
  # ----------------------------------------------------------
  # Holm correction across pairwise comparisons
  # ----------------------------------------------------------
  
  results <- results %>%
    mutate(
      adj_p_value = p.adjust(
        p_value,
        method = "holm"
      )
    )
  
  return(results)
}

auc_pairwise_lesion <- map_dfr(
  unique(dat$cohort),
  function(cc) {
    
    message("Running cohort: ", cc)
    
    d_cc <- dat %>%
      filter(cohort == cc)
    
    cluster_boot_all_auc(
      data = d_cc,
      outcome = "MIBC",
      method_map = method_map,
      patient_id = "patient_id",
      B = 10000,
      seed = 1234
    ) %>%
      mutate(
        cohort = cc,
        analysis_level = "Lesion",
        .before = 1
      )
  }
)

auc_pairwise_table <- auc_pairwise_lesion %>%
  mutate(
    AUC1 = sprintf("%.3f", auc1),
    AUC2 = sprintf("%.3f", auc2),
    
    Difference_CI = sprintf(
      "%.3f (%.3f to %.3f)",
      difference,
      ci_lower,
      ci_upper
    ),
    
    p_value_display = case_when(
      p_value < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", p_value)
    ),
    
    adj_p_value_display = case_when(
      adj_p_value < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", adj_p_value)
    ),
    
    Comparison = paste(method1, "vs", method2)
  ) %>%
  select(
    Comparison,
    cohort,
    AUC1,
    AUC2,
    Difference_CI,
    p_value = p_value_display,
    Adj_p_value = adj_p_value_display
  )
auc_pairwise_table


library(dplyr)

dat %>%
  mutate(
    orig_correct = `Original` == truth,
    qwen_correct = Qwen3 == truth
  ) %>%
  group_by(cohort) %>%
  summarise(
    corrected = sum(!orig_correct & qwen_correct),
    new_errors = sum(orig_correct & !qwen_correct),
    net_gain = corrected - new_errors,
    .groups = "drop"
  )

library(dplyr)

dat %>%
  group_by(cohort, patient_id) %>%
  summarise(n_lesions = n(), .groups = "drop") %>%
  group_by(cohort) %>%
  summarise(
    Total_patients = n(),
    Multiple_lesion_patients = sum(n_lesions > 1),
    Percentage = round(100 * Multiple_lesion_patients / Total_patients, 1)
  )



library(dplyr)
library(tidyr)

table_surgical <- data %>%
  distinct(patient_id, cohort, Surgical) %>%
  count(cohort, Surgical) %>%
  group_by(cohort) %>%
  mutate(
    value = sprintf("%d (%.2f%%)", n, 100 * n / sum(n))
  ) %>%
  select(cohort, Surgical, value) %>%
  pivot_wider(names_from = cohort, values_from = value)

table_surgical

library(dplyr)

data %>%
  distinct(patient_id, cohort) %>%
  count(cohort)
data %>%
  distinct(patient_id, cohort, Surgical) %>%
  count(patient_id) %>%
  filter(n > 1)
distinct(patient_id, cohort, Surgical)
data %>%
  filter(patient_id %in%
           (data %>%
              distinct(patient_id, cohort, Surgical) %>%
              count(patient_id) %>%
              filter(n > 1) %>%
              pull(patient_id))) %>%
  arrange(patient_id)
data %>%
  distinct(patient_id, Surgical) %>%
  count(patient_id) %>%
  count(n)
data %>%
  distinct(patient_id, cohort) %>%
  count(cohort)
data %>%
  distinct(patient_id, cohort, Surgical) %>%
  count(patient_id) %>%
  filter(n > 1)
data %>%
  distinct(patient_id, cohort, Surgical) %>%
  count(cohort, patient_id) %>%
  filter(n > 1)
dup <- data %>%
  distinct(patient_id, cohort, Surgical) %>%
  count(cohort, patient_id) %>%
  filter(n > 1)

data %>%
  semi_join(dup, by = c("cohort", "patient_id")) %>%
  arrange(cohort, patient_id, lesion_id) %>%
  select(patient_id, cohort, lesion_id, Surgical)
patient_surgery <- data %>%
  group_by(cohort, patient_id) %>%
  summarise(
    Surgical = ifelse(any(Surgical == "Cystectomy"),
                      "Cystectomy",
                      "TURBT"),
    .groups = "drop"
  )

patient_surgery %>%
  count(cohort, Surgical) %>%
  group_by(cohort) %>%
  mutate(
    Percent = round(100 * n / sum(n), 2)
  )
library(dplyr)
library(tidyr)

table_surgical <- data %>%
  distinct(cohort, patient_id, Surgical) %>%
  count(cohort, Surgical) %>%
  group_by(cohort) %>%
  mutate(
    value = sprintf("%d (%.2f%%)", n, 100 * n / sum(n))
  ) %>%
  select(cohort, Surgical, value) %>%
  pivot_wider(names_from = cohort, values_from = value)

table_surgical
data %>%
  distinct(cohort, patient_id, Surgical) %>%
  count(cohort, patient_id) %>%
  filter(n > 1)
