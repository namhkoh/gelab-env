# page_id: page_eventbrite_45f56b06f31541079045047b6d542613_18
# screenshot: 2024_4_23_19_27_45f56b06f31541079045047b6d542613-20.png
# step_index: 18/21
# task: Open Eventbrite. Search events 'Yoga session' in New York. Filter free events and set date from May 3 to May 6. What is the location of the first promoted event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint backgrounds and structural elements for the UI page

# Overall background (canvas already white, but fill to ensure consistent color)
bg_color = (255, 255, 255)
draw.rectangle([(0, 0), canvas.size], fill=bg_color)

# Top status bar area (approx ~72px tall) - light neutral gray like the screenshot
status_bar_h = 72
status_bar_color = (190, 190, 190)  # light gray
draw.rectangle([(0, 0), (canvas.size[0], status_bar_h)], fill=status_bar_color)

# Header area under status bar (title area). Keep it white but add subtle bottom divider.
header_top = status_bar_h
header_bottom = 200
header_color = (255, 255, 255)
draw.rectangle([(0, header_top), (canvas.size[0], header_bottom)], fill=header_color)

# Subtle bottom divider under header (thin line)
divider_color = (235, 230, 240)  # very light purple/gray
draw.line([(40, header_bottom), (canvas.size[0]-40, header_bottom)], fill=divider_color, width=2)

# Left content card (Start Date / End Date group) as a rounded light card background
card_left_x = 40
card_right_x = canvas.size[0] - 40
card_top = 250
card_bottom = 580
card_radius = 18
card_fill = (250, 248, 255)  # very pale lavender
draw.rounded_rectangle([(card_left_x, card_top), (card_right_x, card_bottom)],
                       radius=card_radius, fill=card_fill)

# Add a very subtle inner highlight at top of that card to give depth
highlight_color = (255, 255, 255, 40)
# simulate highlight with a lighter rounded rectangle strip
draw.rounded_rectangle([(card_left_x+2, card_top+2), (card_right_x-2, card_top+18)],
                       radius=12, fill=(255,255,255))

# Calendar container area background (separate block below the date card)
cal_top = 620
cal_bottom = 1480
cal_left = 40
cal_right = canvas.size[0] - 40
cal_radius = 10
# Keep calendar background white but give it a faint tint to separate from page
cal_bg = (255, 255, 255)
draw.rounded_rectangle([(cal_left, cal_top), (cal_right, cal_bottom)],
                       radius=cal_radius, fill=cal_bg)

# Month row background (a subtle row where month label and chevrons sit)
month_row_h = 72
month_row_top = cal_top + 10
month_row_bottom = month_row_top + month_row_h
month_row_left = cal_left + 24
month_row_right = cal_right - 24
month_bg = (255, 255, 255)
draw.rectangle([(month_row_left, month_row_top), (month_row_right, month_row_bottom)], fill=month_bg)

# Thin separator under month row
draw.line([(cal_left+12, month_row_bottom+10), (cal_right-12, month_row_bottom+10)],
          fill=(245,240,250), width=1)

# Calendar grid area: leave blank (white) but draw very faint horizontal guide lines for structure
grid_top = month_row_bottom + 30
grid_left = cal_left + 24
grid_right = cal_right - 24
row_height = 120
# draw faint separators for rows (only structural, not numbers)
for i in range(0, 6):
    y = grid_top + i * row_height
    # keep lines very faint so they act as separators only
    draw.line([(grid_left, y), (grid_right, y)], fill=(250,248,252), width=1)

# Add a subtle shadow under the calendar block to lift it slightly
shadow_top = cal_bottom
shadow_height = 8
shadow_color = (240, 238, 246)
draw.rectangle([(cal_left+6, shadow_top), (cal_right-6, shadow_top+shadow_height)], fill=shadow_color)

# Right-side small accessory bar near top (visual structure behind icons, no icons drawn)
# (keeps space visually balanced)
accessory_bar_w = 92
accessory_bar_h = 44
accessory_x = canvas.size[0] - accessory_bar_w - 24
accessory_y = header_top + 18
draw.rounded_rectangle([(accessory_x, accessory_y), (accessory_x+accessory_bar_w, accessory_y+accessory_bar_h)],
                       radius=10, fill=(250,250,251))

# Do not draw or overlap the bottom "Apply date range" button area.
# Reserve bottom margin by drawing a faint top separator above the button region (well above button to avoid duplication)
button_area_top = 2768
separator_y = button_area_top - 28
draw.line([(40, separator_y), (canvas.size[0]-40, separator_y)], fill=(245,243,246), width=2)

# Final subtle overall vignette edges (very faint) to match screenshot feel
edge_shade = (250, 250, 251)
# left and right vertical strips
strip_w = 18
draw.rectangle([(0, 0), (strip_w, canvas.size[1])], fill=edge_shade)
draw.rectangle([(canvas.size[0]-strip_w, 0), (canvas.size[0], canvas.size[1])], fill=edge_shade)

# End of structural drawing. The detected icons, buttons and text will be pasted on top.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 50, 71)
    canvas.paste(_c1, (1154, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1154, 0, 1204, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/02_icon_7.29.png
try:
    _c2 = get_crop(2, 58, 63)
    canvas.paste(_c2, (181, 2), _c2)
except Exception:
    pass
layout["7.29"] = [181, 2, 239, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/03_icon_7.29.png
try:
    _c3 = get_crop(3, 59, 63)
    canvas.paste(_c3, (114, 2), _c3)
except Exception:
    pass
layout["7.29"] = [114, 2, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 100, 71)
    canvas.paste(_c4, (1210, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1210, 0, 1310, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 60, 61)
    canvas.paste(_c5, (311, 4), _c5)
except Exception:
    pass
layout["icon_5"] = [311, 4, 371, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 51, 62)
    canvas.paste(_c6, (248, 3), _c6)
except Exception:
    pass
layout["icon_6"] = [248, 3, 299, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 53, 71)
    canvas.paste(_c7, (1318, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1318, 0, 1371, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/08_icon_7.29.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (12, 72), _c8)
except Exception:
    pass
layout["7.29"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 104, 100)
    canvas.paste(_c9, (204, 780), _c9)
except Exception:
    pass
layout["icon_9"] = [204, 780, 308, 880]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 104, 108)
    canvas.paste(_c10, (72, 775), _c10)
except Exception:
    pass
layout["icon_10"] = [72, 775, 176, 883]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/11_icon_What_date.png
try:
    _c11 = get_crop(11, 319, 72)
    canvas.paste(_c11, (558, 111), _c11)
except Exception:
    pass
layout["What_date?"] = [558, 111, 877, 183]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/12_icon_7.29.png
try:
    _c12 = get_crop(12, 91, 61)
    canvas.paste(_c12, (16, 3), _c12)
except Exception:
    pass
layout["7.29"] = [16, 3, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 50, 64)
    canvas.paste(_c13, (382, 2), _c13)
except Exception:
    pass
layout["icon_13"] = [382, 2, 432, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/14_icon_27.png
try:
    _c14 = get_crop(14, 132, 120)
    canvas.paste(_c14, (192, 1364), _c14)
except Exception:
    pass
layout["27"] = [192, 1364, 324, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/15_icon_Next_month.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (846, 620), _c15)
except Exception:
    pass
layout["Next_month"] = [846, 620, 990, 764]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/16_icon_26.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (60, 1364), _c16)
except Exception:
    pass
layout["26"] = [60, 1364, 192, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/17_icon_May_2024.png
try:
    _c17 = get_crop(17, 103, 111)
    canvas.paste(_c17, (466, 774), _c17)
except Exception:
    pass
layout["May_2024"] = [466, 774, 569, 885]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/18_text_Start_Date.png
try:
    _c18 = get_crop(18, 613, 144)
    canvas.paste(_c18, (48, 313), _c18)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 661, 457]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/19_text_End_Date.png
try:
    _c19 = get_crop(19, 638, 114)
    canvas.paste(_c19, (48, 476), _c19)
except Exception:
    pass
layout["End_Date"] = [48, 476, 686, 590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/20_text_May_2024.png
try:
    _c20 = get_crop(20, 201, 60)
    canvas.paste(_c20, (422, 663), _c20)
except Exception:
    pass
layout["May_2024"] = [422, 663, 623, 723]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/21_text_10.png
try:
    _c21 = get_crop(21, 132, 120)
    canvas.paste(_c21, (720, 1004), _c21)
except Exception:
    pass
layout["10"] = [720, 1004, 852, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/22_text_11.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (852, 1004), _c22)
except Exception:
    pass
layout["11"] = [852, 1004, 984, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/23_text_12.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (60, 1124), _c23)
except Exception:
    pass
layout["12"] = [60, 1124, 192, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/24_text_13.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (192, 1124), _c24)
except Exception:
    pass
layout["13"] = [192, 1124, 324, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/25_text_14.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (324, 1124), _c25)
except Exception:
    pass
layout["14"] = [324, 1124, 456, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/26_text_15.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (456, 1124), _c26)
except Exception:
    pass
layout["15"] = [456, 1124, 588, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/27_text_16.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (588, 1124), _c27)
except Exception:
    pass
layout["16"] = [588, 1124, 720, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/28_text_17.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (720, 1124), _c28)
except Exception:
    pass
layout["17"] = [720, 1124, 852, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/29_text_18.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (852, 1124), _c29)
except Exception:
    pass
layout["18"] = [852, 1124, 984, 1244]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/30_text_19.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (60, 1244), _c30)
except Exception:
    pass
layout["19"] = [60, 1244, 192, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/31_text_20.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (192, 1244), _c31)
except Exception:
    pass
layout["20"] = [192, 1244, 324, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/32_text_21.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (324, 1244), _c32)
except Exception:
    pass
layout["21"] = [324, 1244, 456, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/33_text_22.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (456, 1244), _c33)
except Exception:
    pass
layout["22"] = [456, 1244, 588, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/34_text_23.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (588, 1244), _c34)
except Exception:
    pass
layout["23"] = [588, 1244, 720, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/35_text_24.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (720, 1244), _c35)
except Exception:
    pass
layout["24"] = [720, 1244, 852, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/36_text_25.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (852, 1244), _c36)
except Exception:
    pass
layout["25"] = [852, 1244, 984, 1364]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/37_text_28.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (324, 1364), _c37)
except Exception:
    pass
layout["28"] = [324, 1364, 456, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/38_text_29.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (456, 1364), _c38)
except Exception:
    pass
layout["29"] = [456, 1364, 588, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/39_text_30.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (588, 1364), _c39)
except Exception:
    pass
layout["30"] = [588, 1364, 720, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/40_text_31.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (720, 1364), _c40)
except Exception:
    pass
layout["31"] = [720, 1364, 852, 1484]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 884), _c41)
except Exception:
    pass
layout["1"] = [456, 884, 588, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (588, 884), _c42)
except Exception:
    pass
layout["2"] = [588, 884, 720, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (720, 884), _c43)
except Exception:
    pass
layout["3"] = [720, 884, 852, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/44_clickable_4.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 884), _c44)
except Exception:
    pass
layout["4"] = [852, 884, 984, 1004]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/45_clickable_5.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 1004), _c45)
except Exception:
    pass
layout["5"] = [60, 1004, 192, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/46_clickable_6.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 1004), _c46)
except Exception:
    pass
layout["6"] = [192, 1004, 324, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/47_clickable_7.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 1004), _c47)
except Exception:
    pass
layout["7"] = [324, 1004, 456, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/48_clickable_8.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (456, 1004), _c48)
except Exception:
    pass
layout["8"] = [456, 1004, 588, 1124]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/45f56b06f31541079045047b6d542613/step_18_2024_4_23_19_27_45f56b06f31541079045047b6d542613-20/49_clickable_9.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (588, 1004), _c49)
except Exception:
    pass
layout["9"] = [588, 1004, 720, 1124]
