# page_id: page_eventbrite_c8da01368c4649acb64c6c6b41699bc8_07
# screenshot: 2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9.png
# step_index: 7/13
# task: Open Eventbrite. Look up "Animal" events. Filter by events happening next week. Select the first event - who is the organizer?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas
w, h = canvas.size

# Colors
status_bar_color = (212, 212, 212)     # light gray status bar
toolbar_divider = (232, 229, 243)      # very light purple divider
card_bg = (250, 249, 255)              # very subtle purple-tinted card background
section_bg = (255, 255, 255)           # white for section areas
splitter = (233, 230, 239)             # faint separator line
soft_shadow = (244, 243, 247)          # soft shadow strip near bottom

# 1) Status bar area (top)
status_h = 88
draw.rectangle([0, 0, w, status_h], fill=status_bar_color)

# 2) Toolbar area (below status bar) with a subtle divider line
toolbar_top = status_h
toolbar_bottom = toolbar_top + 112  # header / toolbar region height
draw.rectangle([0, toolbar_top, w, toolbar_bottom], fill=section_bg)
# divider line under toolbar
draw.line([(32, toolbar_bottom), (w-32, toolbar_bottom)], fill=toolbar_divider, width=2)

# 3) Large calendar / date selection card background (subtle rounded card)
cal_left, cal_top = 48, toolbar_bottom + 28
cal_right, cal_bottom = w - 48, cal_top + 1100
try:
    draw.rounded_rectangle([cal_left, cal_top, cal_right, cal_bottom],
                           radius=18, fill=card_bg)
except Exception:
    # Fallback if older PIL doesn't support rounded_rectangle
    draw.rectangle([cal_left, cal_top, cal_right, cal_bottom], fill=card_bg)
# soft inner horizontal separator below calendar header area (visual grouping)
draw.line([(cal_left + 20, cal_top + 120), (cal_right - 20, cal_top + 120)], fill=splitter, width=1)

# 4) Subtle vertical rule to visually separate month heading area (non-intrusive)
draw.line([(w//2, cal_top + 40), (w//2, cal_top + 240)], fill=toolbar_divider, width=1)

# 5) Separator above End Date section
end_sep_y = 1420
draw.line([(32, end_sep_y), (w-32, end_sep_y)], fill=splitter, width=1)

# 6) End Date section background (rounded subtle area behind the heading/content)
end_left, end_top = 48, end_sep_y + 20
end_right, end_bottom = w - 48, end_top + 320
try:
    draw.rounded_rectangle([end_left, end_top, end_right, end_bottom],
                           radius=14, fill=section_bg)
except Exception:
    draw.rectangle([end_left, end_top, end_right, end_bottom], fill=section_bg)
# small divider near top of this section for visual separation
draw.line([(end_left + 12, end_top + 76), (end_right - 12, end_top + 76)], fill=splitter, width=1)

# 7) Large content area background (rest of the screen is white; add a faint band to suggest depth)
content_top = end_bottom + 18
draw.rectangle([0, content_top, w, h], fill=(255,255,255))

# 8) Subtle shadow strip above the bottom control area (do not draw the button itself)
shadow_top = 2660
shadow_bottom = 2760
draw.rectangle([32, shadow_top, w-32, shadow_bottom], fill=soft_shadow)

# 9) Bottom edge subtle boundary to anchor the screen
draw.line([(16, h-38), (w-16, h-38)], fill=splitter, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/00_icon_24.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (456, 1081), _c0)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/03_icon_29.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (192, 1201), _c3)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/04_icon_23.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1081), _c4)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/05_icon_25.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (588, 1081), _c5)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/06_icon_30.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (324, 1201), _c6)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/07_icon_27.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (852, 1081), _c7)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/08_icon_26.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (720, 1081), _c8)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/09_icon_5.15.png
try:
    _c9 = get_crop(9, 61, 65)
    canvas.paste(_c9, (180, 0), _c9)
except Exception:
    pass
layout["5.15"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 62, 64)
    canvas.paste(_c10, (310, 2), _c10)
except Exception:
    pass
layout["icon_10"] = [310, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/11_icon_21.png
try:
    _c11 = get_crop(11, 132, 120)
    canvas.paste(_c11, (60, 1081), _c11)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/12_icon_22.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (192, 1081), _c12)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/13_icon_5.15.png
try:
    _c13 = get_crop(13, 60, 68)
    canvas.paste(_c13, (115, 0), _c13)
except Exception:
    pass
layout["5.15"] = [115, 0, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 64)
    canvas.paste(_c14, (247, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 57, 70)
    canvas.paste(_c15, (1316, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/16_icon_18.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (588, 961), _c16)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/17_icon_5.15.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (12, 72), _c17)
except Exception:
    pass
layout["5.15"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 91, 69)
    canvas.paste(_c18, (1211, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1211, 0, 1302, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/20_icon_19.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (720, 961), _c20)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/21_icon_April_2024.png
try:
    _c21 = get_crop(21, 126, 110)
    canvas.paste(_c21, (593, 611), _c21)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 41, 65)
    canvas.paste(_c22, (1274, 0), _c22)
except Exception:
    pass
layout["icon_22"] = [1274, 0, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 49, 67)
    canvas.paste(_c23, (382, 1), _c23)
except Exception:
    pass
layout["icon_23"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/24_icon_Next_month.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (846, 457), _c24)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/25_icon_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (720, 721), _c25)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/26_icon_12.png
try:
    _c26 = get_crop(26, 104, 107)
    canvas.paste(_c26, (733, 614), _c26)
except Exception:
    pass
layout["12"] = [733, 614, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/27_icon_Choose_a_date.png
try:
    _c27 = get_crop(27, 638, 144)
    canvas.paste(_c27, (48, 1490), _c27)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 104, 100)
    canvas.paste(_c28, (71, 618), _c28)
except Exception:
    pass
layout["icon_28"] = [71, 618, 175, 718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/29_icon_What_date.png
try:
    _c29 = get_crop(29, 322, 71)
    canvas.paste(_c29, (558, 113), _c29)
except Exception:
    pass
layout["What_date?"] = [558, 113, 880, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/30_icon_16.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (324, 961), _c30)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/31_icon_10.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (324, 721), _c31)
except Exception:
    pass
layout["10"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/32_text_5.15.png
try:
    _c32 = get_crop(32, 92, 43)
    canvas.paste(_c32, (22, 17), _c32)
except Exception:
    pass
layout["5.15"] = [22, 17, 114, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/33_text_Start_Date.png
try:
    _c33 = get_crop(33, 589, 114)
    canvas.paste(_c33, (48, 313), _c33)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/34_text_April_2024.png
try:
    _c34 = get_crop(34, 203, 54)
    canvas.paste(_c34, (420, 504), _c34)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/35_text_10.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 841), _c35)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/36_text_11.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 841), _c36)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/37_text_12.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 841), _c37)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/38_text_13.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 841), _c38)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/39_text_14.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (60, 961), _c39)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/40_text_15.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (192, 961), _c40)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/41_text_17.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 961), _c41)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/42_text_20.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 961), _c42)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (192, 721), _c43)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (456, 721), _c44)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/45_clickable_6.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 721), _c45)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/46_clickable_7.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 841), _c46)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/47_clickable_8.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 841), _c47)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c8da01368c4649acb64c6c6b41699bc8/step_07_2024_4_24_17_14_c8da01368c4649acb64c6c6b41699bc8-9/48_clickable_9.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 841), _c48)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
