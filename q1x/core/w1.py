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

def find_monotonic_peaks_left_v2(data_list):
    """左侧单调上升波峰检测"""
    if not data_list:
        return []

    PL = []
    prev = data_list[0]

    for current in data_list[1:]:
        if current > prev:
            prev = current
        elif PL and prev == PL[-1]:  # 关键优化：避免重复添加相同的峰值
            continue
        else:
            PL.append(prev)

    # 处理最后一个元素
    if not PL or prev > PL[-1]:
        PL.append(prev)

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

def detect_main_wave_in_range(data_list):
    """在指定范围内检测主波浪"""
    if len(data_list) == 0:
        return [], [], [], []

    # 前置处理：找出全局最大值（第一个出现的最大值）
    max_val = max(data_list)
    max_idx = data_list.index(max_val)

    # 分割序列
    left_data = data_list[:max_idx + 1]      # [0, max_idx]
    right_data = data_list[max_idx + 1:]     # (max_idx, n-1]

    # 左侧上升浪检测
    PL = find_monotonic_peaks_left(left_data)

    # 右侧上升浪检测
    PR = find_monotonic_peaks_right(right_data)

    # 构建完整主波浪
    all_peaks_with_indices = []

    # 处理PL波峰（从左到右找索引，在左侧数据范围内）
    used_indices = set()
    for peak_val in PL:
        for i in range(len(left_data)):
            if left_data[i] == peak_val:
                global_idx = i  # 左侧数据的索引就是全局索引
                if global_idx not in used_indices:
                    all_peaks_with_indices.append((global_idx, peak_val))
                    used_indices.add(global_idx)
                    break

    # 处理PR波峰（从右到左找索引，在右侧数据范围内）
    for peak_val in PR:
        for i in range(len(right_data) - 1, -1, -1):
            if right_data[i] == peak_val:
                global_idx = max_idx + 1 + i  # 转换为全局索引
                if global_idx not in used_indices:
                    all_peaks_with_indices.append((global_idx, peak_val))
                    used_indices.add(global_idx)
                    break

    # 按索引排序
    all_peaks_with_indices.sort()

    # 构建完整路径
    path_points = []

    # 起始点（第一个元素）
    path_points.append((0, data_list[0]))

    # 添加所有波峰点
    for idx, val in all_peaks_with_indices:
        # 避免重复添加起始点（当起始点恰好也是波峰时）
        if idx != 0:
            path_points.append((idx, val))

    # 在相邻波峰间找波谷
    for i in range(len(all_peaks_with_indices) - 1):
        start_idx, start_val = all_peaks_with_indices[i]
        end_idx, end_val = all_peaks_with_indices[i + 1]

        if end_idx > start_idx + 1:
            # 开区间 (start_idx, end_idx) - 不包含端点
            between_section = data_list[start_idx + 1:end_idx]
            if between_section:
                min_val = min(between_section)
                # 找到最小值的索引（在开区间内查找）
                for j in range(start_idx + 1, end_idx):
                    if data_list[j] == min_val:
                        path_points.append((j, min_val))
                        break

    # 结束点（最后一个元素）
    # 检查是否已存在
    end_idx = len(data_list) - 1
    end_point = (end_idx, data_list[-1])
    if end_point not in path_points:
        path_points.append(end_point)

    # 按索引排序
    path_points.sort(key=lambda x: x[0])

    # 提取峰值
    main_peaks = PL + PR

    # 提取谷值（按照规则）
    valley_values = []

    if len(all_peaks_with_indices) > 0:
        # 1. 第一个谷值：第一个波峰左侧找最小值
        first_peak_idx, first_peak_val = all_peaks_with_indices[0]
        if first_peak_idx > 0:
            left_section = data_list[:first_peak_idx]
            if left_section:
                min_val = min(left_section)
                valley_values.append(min_val)

        # 2. 中间谷值：相邻波峰间找最小值
        for i in range(len(all_peaks_with_indices) - 1):
            start_idx, start_val = all_peaks_with_indices[i]
            end_idx, end_val = all_peaks_with_indices[i + 1]

            if end_idx > start_idx + 1:
                between_section = data_list[start_idx + 1:end_idx]
                if between_section:
                    min_val = min(between_section)
                    # 验证确实是波谷（小于两个相邻波峰）
                    if min_val < start_val and min_val < end_val:
                        valley_values.append(min_val)

        # 3. 最后一个谷值：最后一个波峰右侧找最小值
        last_peak_idx, last_peak_val = all_peaks_with_indices[-1]
        if last_peak_idx < len(data_list) - 1:
            right_section = data_list[last_peak_idx + 1:]
            if right_section:
                min_val = min(right_section)
                valley_values.append(min_val)

    # 构建波浪段
    wave_segments = []
    for i in range(len(path_points) - 1):
        start_idx, start_val = path_points[i]
        end_idx, end_val = path_points[i + 1]
        # 避免无效波浪段（起始点和结束点相同）
        if start_idx != end_idx:
            wave_segments.append((start_idx, end_idx, 0))

    return wave_segments, path_points, main_peaks, valley_values

def detect_wave_recursive(data, start_idx, end_idx, level=0):
    """
    递归检测波浪结构（浪套浪）
    :param data 数据列表
    :param start_idx: 起始索引
    :param end_idx: 结束索引
    :param level: 波浪层级
    :return: 波浪段列表 [(start_idx, end_idx, level), ...]
    """
    # 退出条件：区间太小或无法形成波浪
    if end_idx - start_idx < 3:
        return []

    # 获取子区间数据
    subsection = data[start_idx:end_idx + 1]

    # 在当前区间内检测主波浪
    segments, points, peaks, valleys = detect_main_wave_in_range(subsection)

    # 退出条件：没有检测到波浪结构
    if not peaks and not valleys:
        return []

    # 转换为全局索引的波浪段
    global_segments = []

    # 构建当前层级的波浪段
    for i in range(len(points) - 1):
        local_start_idx, start_val = points[i]
        local_end_idx, end_val = points[i + 1]
        global_start = start_idx + local_start_idx
        global_end = start_idx + local_end_idx

        # 判断是上升还是下降
        is_rising = end_val > start_val
        global_segments.append((global_start, global_end, level, is_rising))

    # 递归检测相邻波峰间的次级波浪
    for i in range(len(points) - 1):
        local_peak1_idx, _ = points[i]
        local_peak2_idx, _ = points[i + 1]
        global_peak1 = start_idx + local_peak1_idx
        global_peak2 = start_idx + local_peak2_idx

        # 在两个波峰间的开区间递归（确保有足够的点形成波浪）
        if global_peak2 - global_peak1 >= 3:
            sub_waves = detect_wave_recursive(data, global_peak1, global_peak2, level + 1)
            global_segments.extend(sub_waves)

    return global_segments

def detect_complete_wave_structure(data):
    """检测完整的波浪结构（包括主波浪和次级波浪）"""
    print("=== 检测完整波浪结构 ===")
    data_list = data.tolist() if isinstance(data, np.ndarray) else list(data)
    print(f"数据: {data_list}")
    print(f"索引: {list(range(len(data_list)))}")

    if len(data_list) < 3:
        return []

    # 首先检测主波浪结构
    main_segments, main_points, main_peaks, main_valleys = detect_main_wave_in_range(data_list)

    print(f"\n主波浪结构:")
    print(f"峰值: {main_peaks}")
    print(f"谷值: {main_valleys}")

    # 转换主波浪段为全局索引并标记为Level 0
    all_wave_segments = []
    for i in range(len(main_points) - 1):
        start_idx, start_val = main_points[i]
        end_idx, end_val = main_points[i + 1]
        if start_idx != end_idx:
            is_rising = end_val > start_val
            all_wave_segments.append((start_idx, end_idx, 0, is_rising))
            print(f"主波浪段: 索引{start_idx} -> {end_idx} (Level 0, {'上升' if is_rising else '下降'})")

    # 在相邻波峰间递归检测次级波浪
    for i in range(len(main_points) - 1):
        start_peak_idx, start_val = main_points[i]
        end_peak_idx, end_val = main_points[i + 1]

        # 在开区间内递归检测
        if end_peak_idx - start_peak_idx >= 3:
            sub_waves = detect_wave_recursive(data_list, start_peak_idx, end_peak_idx, 1)
            all_wave_segments.extend(sub_waves)

    # 按层级和索引排序
    all_wave_segments.sort(key=lambda x: (x[2], x[0]))

    print(f"\n完整波浪结构:")
    for i, (start, end, level, is_rising) in enumerate(all_wave_segments):
        direction = "上升" if is_rising else "下降"
        print(f"  段{i+1}: 索引{start} -> {end} (Level {level}, {direction})")

    return all_wave_segments

def plot_wave_structure(data, wave_segments):
    """绘制波浪结构图 - 优化版本"""
    plt.figure(figsize=(14, 8))
    data_list = data.tolist() if isinstance(data, np.ndarray) else list(data)
    x = list(range(len(data_list)))

    # 绘制原始数据
    plt.plot(x, data_list, 'k-', linewidth=1, alpha=0.7, label='原始数据')
    plt.scatter(x, data_list, c='black', s=20, alpha=0.7)

    # 颜色定义
    # 主波浪颜色
    primary_rising_color = 'red'      # 波谷到波峰用红色
    primary_falling_color = 'green'   # 波峰到波谷用绿色

    # 次级波浪颜色（变淡）
    secondary_rising_color = '#ff9999'   # 淡红色
    secondary_falling_color = '#99cc99'  # 淡绿色

    # 更深层波浪颜色（更淡）
    tertiary_rising_color = '#ffcccc'    # 更淡红色
    tertiary_falling_color = '#ccffcc'   # 更淡绿色

    def get_color(level, is_rising):
        """根据层级和方向获取颜色"""
        if level == 0:  # 主波浪
            return primary_rising_color if is_rising else primary_falling_color
        elif level == 1:  # 次级波浪
            return secondary_rising_color if is_rising else secondary_falling_color
        else:  # 更深层波浪
            return tertiary_rising_color if is_rising else tertiary_falling_color

    def get_alpha(level):
        """根据层级获取透明度"""
        if level == 0:
            return 1.0
        elif level == 1:
            return 0.8
        elif level == 2:
            return 0.6
        else:
            return 0.4

    def get_linewidth(level):
        """根据层级获取线宽"""
        if level == 0:
            return 3.0
        elif level == 1:
            return 2.0
        elif level == 2:
            return 1.5
        else:
            return 1.0

    # 按层级分组波浪段
    level_segments = {}
    for segment in wave_segments:
        start_idx, end_idx, level, is_rising = segment
        if level not in level_segments:
            level_segments[level] = []
        level_segments[level].append(segment)

    # 绘制波浪段
    for level in sorted(level_segments.keys()):
        segments = level_segments[level]

        for i, (start_idx, end_idx, lvl, is_rising) in enumerate(segments):
            start_val = data_list[start_idx]
            end_val = data_list[end_idx]

            # 获取颜色、透明度和线宽
            color = get_color(lvl, is_rising)
            alpha = get_alpha(lvl)
            linewidth = get_linewidth(lvl)

            # 绘制波浪线
            plt.plot([start_idx, end_idx], [start_val, end_val],
                     color=color, linewidth=linewidth, alpha=alpha)

            # 添加标签（只为主波浪添加避免混乱）
            if level == 0:
                mid_x = (start_idx + end_idx) / 2
                mid_y = (start_val + end_val) / 2
                direction = "↑" if is_rising else "↓"
                plt.annotate(f'L{level}{direction}', (mid_x, mid_y),
                             fontsize=9, color=color, alpha=alpha)

    plt.title('波浪检测结果 - 大浪套小浪结构\n(红色↑:波谷到波峰, 绿色↓:波峰到波谷)', fontsize=14)
    plt.xlabel('时间索引')
    plt.ylabel('数值')
    plt.grid(True, alpha=0.3)

    # 创建图例
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], color=primary_rising_color, linewidth=3, label='Level 0 ↑ (波谷→波峰)'))
    legend_elements.append(plt.Line2D([0], [0], color=primary_falling_color, linewidth=3, label='Level 0 ↓ (波峰→波谷)'))
    legend_elements.append(plt.Line2D([0], [0], color=secondary_rising_color, linewidth=2, label='Level 1 ↑ (波谷→波峰)'))
    legend_elements.append(plt.Line2D([0], [0], color=secondary_falling_color, linewidth=2, label='Level 1 ↓ (波峰→波谷)'))
    legend_elements.append(plt.Line2D([0], [0], color=tertiary_rising_color, linewidth=1.5, label='Level 2 ↑ (波谷→波峰)'))
    legend_elements.append(plt.Line2D([0], [0], color=tertiary_falling_color, linewidth=1.5, label='Level 2 ↓ (波峰→波谷)'))

    plt.legend(handles=legend_elements, loc='upper right')

    # 显示统计信息
    print("\n波浪段统计:")
    for level in sorted(level_segments.keys()):
        segments = level_segments[level]
        rising_count = sum(1 for seg in segments if seg[3])  # is_rising
        falling_count = len(segments) - rising_count
        print(f"Level {level}: 总计{len(segments)}段 (↑{rising_count}段, ↓{falling_count}段)")

    plt.tight_layout()
    plt.show()

# 测试
if __name__ == "__main__":
    # 测试数据
    data = np.array([1, 9, 1, 5, 3, 4, 2, 7, 4, 6, 2, 9, 4])

    # 检测完整波浪结构
    complete_waves = detect_complete_wave_structure(data)

    print(f"\n=== 最终结果 ===")
    for i, (start, end, level, is_rising) in enumerate(complete_waves):
        direction = "上升" if is_rising else "下降"
        print(f"波浪段{i+1}: 索引{start} -> {end} (Level {level}, {direction})")

    # 绘制结果
    plot_wave_structure(data, complete_waves)