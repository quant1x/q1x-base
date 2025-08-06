import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def find_monotonic_peaks_left(data_list):
    """左侧单调上升波峰检测"""
    if len(data_list) == 0:
        return []
    PL = []
    Plast = data_list[0]
    last_added = False
    for i in range(1, len(data_list)):
        current = data_list[i]
        if current > Plast:
            Plast = current
            last_added = False
        else:
            if not last_added:
                PL.append(Plast)
                last_added = True
    if not last_added:
        if len(PL) == 0 or Plast > PL[-1]:
            PL.append(Plast)
    return PL


def find_monotonic_peaks_right(data_list):
    """右侧单调上升波峰检测（从右向左）"""
    if len(data_list) == 0:
        return []
    PR = []
    Prast = data_list[-1]
    last_added = False
    for i in range(len(data_list) - 2, -1, -1):
        current = data_list[i]
        if current > Prast:
            Prast = current
            last_added = False
        else:
            if not last_added:
                PR.append(Prast)
                last_added = True
    if not last_added:
        if len(PR) == 0 or Prast > PR[-1]:
            PR.append(Prast)
    PR.reverse()
    return PR


def detect_main_wave_in_range(high_list, low_list):
    """在指定范围内检测主波浪，波峰来自high_list，波谷来自low_list"""
    if len(high_list) == 0 or len(low_list) == 0:
        return [], [], [], []

    # 使用高价序列找全局最大值作为分割点
    max_val = max(high_list)
    max_idx = high_list.index(max_val)

    # 左侧高价数据
    left_high = high_list[:max_idx + 1]
    # 右侧高价数据
    right_high = high_list[max_idx + 1:]

    # 检测高价序列的波峰
    PL = find_monotonic_peaks_left(left_high)
    PR = find_monotonic_peaks_right(right_high)

    # 合并所有高价波峰及其全局索引
    all_peaks_with_indices = []
    used_indices = set()

    # 处理左侧波峰
    for peak_val in PL:
        for i in range(len(left_high)):
            if left_high[i] == peak_val:
                global_idx = i
                if global_idx not in used_indices:
                    all_peaks_with_indices.append((global_idx, peak_val))
                    used_indices.add(global_idx)
                    break

    # 处理右侧波峰
    for peak_val in PR:
        for i in range(len(right_high) - 1, -1, -1):
            if right_high[i] == peak_val:
                global_idx = max_idx + 1 + i
                if global_idx not in used_indices:
                    all_peaks_with_indices.append((global_idx, peak_val))
                    used_indices.add(global_idx)
                    break

    all_peaks_with_indices.sort()

    # 构建路径点（高价点）
    path_points = []
    if high_list:
        path_points.append((0, high_list[0]))

    for idx, val in all_peaks_with_indices:
        if idx != 0:
            path_points.append((idx, val))

    # 在相邻高价波峰之间，从低价序列中找波谷
    for i in range(len(all_peaks_with_indices) - 1):
        start_idx, _ = all_peaks_with_indices[i]
        end_idx, _ = all_peaks_with_indices[i + 1]
        if end_idx > start_idx + 1:
            between_low = low_list[start_idx + 1:end_idx]
            if between_low:
                min_val = min(between_low)
                for j in range(start_idx + 1, end_idx):
                    if low_list[j] == min_val:
                        path_points.append((j, min_val))
                        break

    # 添加结束点（高价序列的最后一个点）
    end_idx = len(high_list) - 1
    end_point = (end_idx, high_list[-1])
    if end_point not in path_points:
        path_points.append(end_point)

    path_points.sort(key=lambda x: x[0])

    # 提取主波峰（来自高价）
    main_peaks = PL + PR

    # 提取波谷（来自低价）
    valley_values = []
    if all_peaks_with_indices:
        first_peak_idx, _ = all_peaks_with_indices[0]
        if first_peak_idx > 0:
            left_low = low_list[:first_peak_idx]
            if left_low:
                valley_values.append(min(left_low))

        for i in range(len(all_peaks_with_indices) - 1):
            start_idx, _ = all_peaks_with_indices[i]
            end_idx, _ = all_peaks_with_indices[i + 1]
            if end_idx > start_idx + 1:
                between_low = low_list[start_idx + 1:end_idx]
                if between_low:
                    min_val = min(between_low)
                    # 可加验证：是否低于相邻高价波峰
                    valley_values.append(min_val)

        last_peak_idx, _ = all_peaks_with_indices[-1]
        if last_peak_idx < len(low_list) - 1:
            right_low = low_list[last_peak_idx + 1:]
            if right_low:
                valley_values.append(min(right_low))

    # 构建波浪段
    wave_segments = []
    for i in range(len(path_points) - 1):
        start_idx, start_val = path_points[i]
        end_idx, end_val = path_points[i + 1]
        if start_idx != end_idx:
            is_rising = end_val > start_val
            wave_segments.append((start_idx, end_idx, 0, is_rising))

    return wave_segments, path_points, main_peaks, valley_values


def detect_wave_recursive(high_data, low_data, start_idx, end_idx, level=0):
    """
    递归检测波浪结构（浪套浪），使用高价和低价序列
    """
    if end_idx - start_idx < 3:
        return []

    high_sub = high_data[start_idx:end_idx + 1]
    low_sub = low_data[start_idx:end_idx + 1]

    segments, points, peaks, valleys = detect_main_wave_in_range(high_sub, low_sub)
    if not peaks and not valleys:
        return []

    global_segments = []
    for i in range(len(points) - 1):
        local_start_idx, start_val = points[i]
        local_end_idx, end_val = points[i + 1]
        global_start = start_idx + local_start_idx
        global_end = start_idx + local_end_idx
        if global_start != global_end:
            is_rising = end_val > start_val
            global_segments.append((global_start, global_end, level, is_rising))

    for i in range(len(points) - 1):
        local_peak1_idx, _ = points[i]
        local_peak2_idx, _ = points[i + 1]
        global_peak1 = start_idx + local_peak1_idx
        global_peak2 = start_idx + local_peak2_idx
        if global_peak2 - global_peak1 >= 3:
            sub_waves = detect_wave_recursive(high_data, low_data, global_peak1, global_peak2, level + 1)
            global_segments.extend(sub_waves)

    return global_segments


def detect_complete_wave_structure(high_data, low_data):
    """检测完整的波浪结构（包括主波浪和次级波浪）"""
    print("=== 检测完整波浪结构 ===")
    high_list = high_data.tolist() if isinstance(high_data, np.ndarray) else list(high_data)
    low_list = low_data.tolist() if isinstance(low_data, np.ndarray) else list(low_data)

    print(f"高价序列: {high_list}")
    print(f"低价序列: {low_list}")
    print(f"索引: {list(range(len(high_list)))}")

    if len(high_list) < 3:
        return []

    main_segments, main_points, main_peaks, main_valleys = detect_main_wave_in_range(high_list, low_list)

    print(f"\n主波浪结构:")
    print(f"高价波峰: {main_peaks}")
    print(f"低价波谷: {main_valleys}")

    all_wave_segments = []
    for i in range(len(main_points) - 1):
        start_idx, start_val = main_points[i]
        end_idx, end_val = main_points[i + 1]
        if start_idx != end_idx:
            is_rising = end_val > start_val
            all_wave_segments.append((start_idx, end_idx, 0, is_rising))
            print(f"主波浪段: 索引{start_idx} -> {end_idx} (Level 0, {'上升' if is_rising else '下降'})")

    for i in range(len(main_points) - 1):
        start_peak_idx, _ = main_points[i]
        end_peak_idx, _ = main_points[i + 1]
        if end_peak_idx - start_peak_idx >= 3:
            sub_waves = detect_wave_recursive(high_list, low_list, start_peak_idx, end_peak_idx, 1)
            all_wave_segments.extend(sub_waves)

    all_wave_segments.sort(key=lambda x: (x[2], x[0]))

    print(f"\n完整波浪结构:")
    for i, (start, end, level, is_rising) in enumerate(all_wave_segments):
        direction = "上升" if is_rising else "下降"
        print(f"  段{i+1}: 索引{start} -> {end} (Level {level}, {direction})")

    return all_wave_segments


def plot_wave_structure(high_data, low_data, wave_segments):
    """绘制波浪结构图 - 支持高价和低价序列，并添加水印"""
    plt.figure(figsize=(14, 8))
    high_list = high_data.tolist() if isinstance(high_data, np.ndarray) else list(high_data)
    low_list = low_data.tolist() if isinstance(low_data, np.ndarray) else list(low_data)
    x = list(range(len(high_list)))

    # 绘制原始高价和低价
    plt.plot(x, high_list, 'k-', linewidth=1, alpha=0.7, label='高价序列')
    plt.plot(x, low_list, 'gray', linewidth=1, alpha=0.7, label='低价序列')
    plt.scatter(x, high_list, c='black', s=20, alpha=0.7)
    plt.scatter(x, low_list, c='gray', s=20, alpha=0.7)

    # 颜色定义
    primary_rising_color = 'red'
    primary_falling_color = 'green'
    secondary_rising_color = '#ff9999'
    secondary_falling_color = '#99cc99'
    tertiary_rising_color = '#ffcccc'
    tertiary_falling_color = '#ccffcc'

    def get_color(level, is_rising):
        if level == 0:
            return primary_rising_color if is_rising else primary_falling_color
        elif level == 1:
            return secondary_rising_color if is_rising else secondary_falling_color
        else:
            return tertiary_rising_color if is_rising else tertiary_falling_color

    def get_alpha(level):
        return 1.0 if level == 0 else 0.8 if level == 1 else 0.6 if level == 2 else 0.4

    def get_linewidth(level):
        return 3.0 if level == 0 else 2.0 if level == 1 else 1.5 if level == 2 else 1.0

    level_segments = {}
    for seg in wave_segments:
        level = seg[2]
        if level not in level_segments:
            level_segments[level] = []
        level_segments[level].append(seg)

    for level in sorted(level_segments.keys()):
        for start_idx, end_idx, lvl, is_rising in level_segments[level]:
            start_val = high_list[start_idx] if (start_idx, high_list[start_idx]) in [(p[0], p[1]) for p in detect_main_wave_in_range(high_list, low_list)[1]] else low_list[start_idx]
            end_val = high_list[end_idx] if (end_idx, high_list[end_idx]) in [(p[0], p[1]) for p in detect_main_wave_in_range(high_list, low_list)[1]] else low_list[end_idx]
            color = get_color(lvl, is_rising)
            alpha = get_alpha(lvl)
            linewidth = get_linewidth(lvl)
            plt.plot([start_idx, end_idx], [start_val, end_val], color=color, linewidth=linewidth, alpha=alpha)

    plt.title('波浪检测结果 - 基于高低价序列\n(红色↑:波谷→波峰, 绿色↓:波峰→波谷)', fontsize=14)
    plt.xlabel('时间索引')
    plt.ylabel('价格')
    plt.grid(True, alpha=0.3)

    legend_elements = [
        plt.Line2D([0], [0], color='k', linewidth=1, label='高价序列'),
        plt.Line2D([0], [0], color='gray', linewidth=1, label='低价序列'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Level 0 ↑'),
        plt.Line2D([0], [0], color='green', linewidth=3, label='Level 0 ↓'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # -------------------------------
    # ✅ 添加45度倾斜水印
    # -------------------------------
    fig = plt.gcf()  # 获取当前figure
    ax = plt.gca()   # 获取当前axes

    # 创建一个覆盖整个图表的透明轴用于显示水印
    watermark_ax = fig.add_subplot(111, facecolor='none')
    watermark_ax.patch.set_alpha(0.0)  # 完全透明

    # 关闭坐标轴
    watermark_ax.set_xticks([])
    watermark_ax.set_yticks([])
    watermark_ax.spines['top'].set_visible(False)
    watermark_ax.spines['bottom'].set_visible(False)
    watermark_ax.spines['left'].set_visible(False)
    watermark_ax.spines['right'].set_visible(False)

    # 设置水印文字
    watermark_text = "Quant1X"
    angle = 45      # 倾斜角度
    alpha = 0.1     # 透明度 (0-1, 越小越淡)
    fontsize = 100  # 字体大小

    watermark_ax.text(
        0.5, 0.5, watermark_text,
        rotation=angle,
        alpha=alpha,
        fontsize=fontsize,
        color='gray',
        ha='center',
        va='center',
        transform=watermark_ax.transAxes, # 使用axes坐标系
        zorder=-100 # 确保在最底层
    )
    # -------------------------------
    # 水印添加结束
    # -------------------------------

    plt.tight_layout()
    plt.show()


# 测试
if __name__ == "__main__":
    # 示例：高价和低价序列（长度相同）
    high_data = np.array([1, 10, 2, 6, 4, 5, 3, 8, 5, 7, 3, 10, 5])
    low_data = np.array([0, 8, 0, 4, 2, 3, 1, 6, 3, 5, 1, 8, 3])

    complete_waves = detect_complete_wave_structure(high_data, low_data)
    plot_wave_structure(high_data, low_data, complete_waves)