# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_14
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16.png
# step_index: 14/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# background & structural layout for the calendar UI
# assumes: canvas (1440x2960 RGB PIL Image) and draw (ImageDraw) and fonts are available

# Fill overall background (dominant color = white)
draw.rectangle((0, 0, 1440, 2960), fill=(255, 255, 255))

# Status bar area at top (~72px)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(200, 200, 200))  # light neutral gray for status bar
# subtle bottom border for status bar
draw.line([(0, status_h), (1440, status_h)], fill=(180, 180, 180), width=1)

# Header / toolbar area (under status bar)
header_top = status_h
header_bottom = 216
draw.rectangle((0, header_top, 1440, header_bottom), fill=(255, 255, 255))  # keep header background white
# header bottom divider
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill=(235, 233, 240), width=2)

# Calendar card background (rounded rectangle) - behind the month label and the date grid
cal_x0, cal_y0 = 48, 600
cal_x1, cal_y1 = 1392, 1520
cal_radius = 28
draw.rounded_rectangle([cal_x0, cal_y0, cal_x1, cal_y1],
                       radius=cal_radius,
                       fill=(250, 250, 253),     # very subtle off-white to separate from page white
                       outline=(230, 228, 238),
                       width=2)

# Slight top accent line inside calendar card (to separate month header from grid)
month_header_y = cal_y0 + 56
draw.line([(cal_x0 + 28, month_header_y), (cal_x1 - 28, month_header_y)],
          fill=(240, 238, 245), width=1)

# Weekday row separator (very faint) where weekday labels sit
weekday_row_y = month_header_y + 64
draw.line([(cal_x0 + 28, weekday_row_y), (cal_x1 - 28, weekday_row_y)],
          fill=(245, 244, 247), width=1)

# Subtle grid background band for the calendar numbers area
grid_top = weekday_row_y + 12
grid_bottom = cal_y1 - 24
draw.rectangle((cal_x0 + 16, grid_top, cal_x1 - 16, grid_bottom), fill=(255, 255, 255))

# A light shadow line under the calendar card to give slight lift
shadow_y = cal_y1 + 6
draw.line([(cal_x0 + 8, shadow_y), (cal_x1 - 8, shadow_y)], fill=(240, 239, 244), width=2)

# Large whitespace content area below the calendar (just a subtle band to separate bottom region)
content_sep_y = 1660
draw.line([(24, content_sep_y), (1440-24, content_sep_y)], fill=(246, 246, 248), width=1)

# Separator above the bottom "Apply date range" control (do not draw the control itself)
# The detected button area starts at y=2768, so draw a separator comfortably above it
apply_sep_y = 2720
draw.line([(24, apply_sep_y), (1440-24, apply_sep_y)], fill=(230, 228, 236), width=2)

# Light inset border to show safe zone for content (subtle)
inset_margin = 24
draw.rectangle([inset_margin, header_bottom + 12, 1440 - inset_margin, apply_sep_y - 24],
               outline=(245, 244, 247), width=1)

# Decorative thin vertical guide at left to align form elements (very faint)
draw.line([(48, header_bottom + 24), (48, apply_sep_y - 24)], fill=(250, 249, 252), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 50, 71)
    canvas.paste(_c1, (1154, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1154, 0, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/02_icon_7.48.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.48"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/03_icon_7.48.png
try:
    _c3 = get_crop(3, 59, 65)
    canvas.paste(_c3, (115, 1), _c3)
except Exception:
    pass
layout["7.48"] = [115, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 61, 62)
    canvas.paste(_c4, (310, 3), _c4)
except Exception:
    pass
layout["icon_4"] = [310, 3, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 100, 71)
    canvas.paste(_c5, (1210, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1210, 0, 1310, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/06_icon_22.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (456, 1244), _c6)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 106, 108)
    canvas.paste(_c7, (71, 775), _c7)
except Exception:
    pass
layout["icon_7"] = [71, 775, 177, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/08_icon_27.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (192, 1364), _c8)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 60)
    canvas.paste(_c9, (249, 5), _c9)
except Exception:
    pass
layout["icon_9"] = [249, 5, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 70)
    canvas.paste(_c10, (1318, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1318, 0, 1371, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 116, 110)
    canvas.paste(_c11, (202, 774), _c11)
except Exception:
    pass
layout["icon_11"] = [202, 774, 318, 884]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/12_icon_7.48.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (12, 72), _c12)
except Exception:
    pass
layout["7.48"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/13_icon_28.png
try:
    _c13 = get_crop(13, 132, 120)
    canvas.paste(_c13, (324, 1364), _c13)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/14_icon_26.png
try:
    _c14 = get_crop(14, 132, 120)
    canvas.paste(_c14, (60, 1364), _c14)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/15_icon_May_2024.png
try:
    _c15 = get_crop(15, 134, 110)
    canvas.paste(_c15, (326, 773), _c15)
except Exception:
    pass
layout["May_2024"] = [326, 773, 460, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/16_icon_7.48.png
try:
    _c16 = get_crop(16, 90, 62)
    canvas.paste(_c16, (17, 3), _c16)
except Exception:
    pass
layout["7.48"] = [17, 3, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/17_icon_23.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (588, 1244), _c17)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/18_icon_What_date.png
try:
    _c18 = get_crop(18, 319, 73)
    canvas.paste(_c18, (558, 111), _c18)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 50, 65)
    canvas.paste(_c19, (382, 2), _c19)
except Exception:
    pass
layout["icon_19"] = [382, 2, 432, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/20_icon_10.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (588, 884), _c20)
except Exception:
    pass
layout["10"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/21_icon_May_2024.png
try:
    _c21 = get_crop(21, 108, 107)
    canvas.paste(_c21, (463, 776), _c21)
except Exception:
    pass
layout["May_2024"] = [463, 776, 571, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/22_icon_15.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (456, 1124), _c22)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/23_icon_Next_month.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (846, 620), _c23)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/24_icon_29.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (456, 1364), _c24)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/25_icon_24.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (720, 1244), _c25)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/26_text_Start_Date.png
try:
    _c26 = get_crop(26, 591, 144)
    canvas.paste(_c26, (48, 313), _c26)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 639, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/27_text_End_Date.png
try:
    _c27 = get_crop(27, 581, 114)
    canvas.paste(_c27, (48, 476), _c27)
except Exception:
    pass
layout["End_Date"] = [48, 476, 629, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/28_text_May_2024.png
try:
    _c28 = get_crop(28, 198, 56)
    canvas.paste(_c28, (423, 666), _c28)
except Exception:
    pass
layout["May_2024"] = [423, 666, 621, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/29_text_10.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (720, 1004), _c29)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/30_text_11.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (852, 1004), _c30)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/31_text_12.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (60, 1124), _c31)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/32_text_13.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (192, 1124), _c32)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/33_text_14.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (324, 1124), _c33)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/34_text_16.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (588, 1124), _c34)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/35_text_17.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (720, 1124), _c35)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/36_text_18.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (852, 1124), _c36)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/37_text_19.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (60, 1244), _c37)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/38_text_20.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (192, 1244), _c38)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/39_text_21.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (324, 1244), _c39)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/40_text_25.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (852, 1244), _c40)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/41_text_30.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (588, 1364), _c41)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/42_text_31.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (720, 1364), _c42)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 884), _c43)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (720, 884), _c44)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/45_clickable_4.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 884), _c45)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/46_clickable_5.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 1004), _c46)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/47_clickable_6.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 1004), _c47)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/48_clickable_7.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 1004), _c48)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/49_clickable_8.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (456, 1004), _c49)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_14_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-16/50_clickable_9.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (588, 1004), _c50)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
