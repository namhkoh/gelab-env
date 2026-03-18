# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_11
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13.png
# step_index: 11/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page.
# Uses provided variables: canvas (PIL Image) and draw (ImageDraw), and fonts.

# Colors
bg_color = "#FFFFFF"
status_bar_color = "#BDBDBD"       # top status bar gray
header_divider = "#EDE7F2"         # pale purple divider
card_bg = "#FBF9FF"                # very subtle off-white/purple card background
card_outline = "#E9E3EE"           # card outline
separator_color = "#EDE7F2"        # section separators
shadow_color = "#E6E0E8"           # subtle shadow/edge

w, h = canvas.size

# Base background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area)
status_h = 80
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header / toolbar area (below status bar)
header_top = status_h
header_bottom = 200
draw.rectangle([(0, header_top), (w, header_bottom)], fill=bg_color)
# header bottom divider
draw.line([(24, header_bottom), (w - 24, header_bottom)], fill=header_divider, width=2)

# Calendar / Start Date card background (rounded rectangle)
card_left = 48
card_top = 200
card_right = w - 48
card_bottom = 1500
radius = 28
draw.rounded_rectangle(
    [(card_left, card_top), (card_right, card_bottom)],
    radius=radius,
    fill=card_bg,
    outline=card_outline,
    width=1
)
# slight top shadow line for card
draw.line([(card_left + 2, card_top + 2), (card_right - 2, card_top + 2)], fill=shadow_color, width=1)

# Separator between Start Date and End Date sections
sep_y = 1488
draw.line([(48, sep_y), (w - 48, sep_y)], fill=separator_color, width=2)

# End Date section card background (sub-card, lighter)
end_card_top = 1508
end_card_bottom = 2600
draw.rounded_rectangle(
    [(card_left, end_card_top), (card_right, end_card_bottom)],
    radius=20,
    fill=card_bg,
    outline=card_outline,
    width=1
)
# subtle divider line near top of end section
draw.line([(card_left + 8, end_card_top + 8), (card_right - 8, end_card_top + 8)], fill=shadow_color, width=1)

# Decorative thin separators to suggest content grouping (no text/icons)
# A few light horizontal guides for calendar week separation (background only)
week_start_y = card_top + 160
week_height = 120
for i in range(1, 5):
    y = week_start_y + i * week_height
    if y < card_bottom - 40:
        draw.line([(card_left + 24, y), (card_right - 24, y)], fill="#F3EFF6", width=1)

# Left and right subtle vertical margins to frame the content
margin_x = 24
draw.line([(margin_x, header_bottom + 6), (margin_x, card_bottom)], fill="#F4F1F6", width=1)
draw.line([(w - margin_x, header_bottom + 6), (w - margin_x, card_bottom)], fill="#F4F1F6", width=1)

# Ensure we do not draw over the bottom "Apply date range" button area (detected element)
# (Apply button area: (48,2768) size 1344x144) - we avoid drawing in that region by design above.

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 52, 71)
    canvas.paste(_c1, (1153, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/02_icon_May.png
try:
    _c2 = get_crop(2, 128, 115)
    canvas.paste(_c2, (195, 610), _c2)
except Exception:
    pass
layout["May"] = [195, 610, 323, 725]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/03_icon_28.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (324, 1201), _c3)
except Exception:
    pass
layout["28"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/04_icon_26.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (60, 1201), _c4)
except Exception:
    pass
layout["26"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/05_icon_7.47.png
try:
    _c5 = get_crop(5, 62, 66)
    canvas.paste(_c5, (179, 1), _c5)
except Exception:
    pass
layout["7.47"] = [179, 1, 241, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/06_icon_7.47.png
try:
    _c6 = get_crop(6, 64, 68)
    canvas.paste(_c6, (111, 0), _c6)
except Exception:
    pass
layout["7.47"] = [111, 0, 175, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/07_icon_May.png
try:
    _c7 = get_crop(7, 139, 115)
    canvas.paste(_c7, (321, 608), _c7)
except Exception:
    pass
layout["May"] = [321, 608, 460, 723]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 65, 65)
    canvas.paste(_c8, (308, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [308, 2, 373, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 101, 70)
    canvas.paste(_c9, (1210, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [1210, 0, 1311, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/10_icon_27.png
try:
    _c10 = get_crop(10, 132, 120)
    canvas.paste(_c10, (192, 1201), _c10)
except Exception:
    pass
layout["27"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/11_icon_May.png
try:
    _c11 = get_crop(11, 130, 115)
    canvas.paste(_c11, (455, 609), _c11)
except Exception:
    pass
layout["May"] = [455, 609, 585, 724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 53, 65)
    canvas.paste(_c12, (247, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [247, 1, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 70)
    canvas.paste(_c13, (1318, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1318, 0, 1372, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/14_icon_29.png
try:
    _c14 = get_crop(14, 132, 120)
    canvas.paste(_c14, (456, 1201), _c14)
except Exception:
    pass
layout["29"] = [456, 1201, 588, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 105, 117)
    canvas.paste(_c15, (71, 611), _c15)
except Exception:
    pass
layout["icon_15"] = [71, 611, 176, 728]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/16_icon_May.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (54, 457), _c16)
except Exception:
    pass
layout["May"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/17_icon_7.47.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (12, 72), _c17)
except Exception:
    pass
layout["7.47"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/18_icon_2024.png
try:
    _c18 = get_crop(18, 129, 113)
    canvas.paste(_c18, (591, 609), _c18)
except Exception:
    pass
layout["2024"] = [591, 609, 720, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/19_icon_7.47.png
try:
    _c19 = get_crop(19, 92, 63)
    canvas.paste(_c19, (16, 2), _c19)
except Exception:
    pass
layout["7.47"] = [16, 2, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 49, 68)
    canvas.paste(_c20, (382, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [382, 1, 431, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/21_icon_Next_month.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (846, 457), _c21)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/22_icon_22.png
try:
    _c22 = get_crop(22, 132, 120)
    canvas.paste(_c22, (456, 1081), _c22)
except Exception:
    pass
layout["22"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/23_icon_23.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (588, 1081), _c23)
except Exception:
    pass
layout["23"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/24_text_What_date.png
try:
    _c24 = get_crop(24, 318, 63)
    canvas.paste(_c24, (563, 117), _c24)
except Exception:
    pass
layout["What_date?"] = [563, 117, 881, 180]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/25_text_Start_Date.png
try:
    _c25 = get_crop(25, 580, 114)
    canvas.paste(_c25, (48, 313), _c25)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 628, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/26_text_10.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (720, 841), _c26)
except Exception:
    pass
layout["10"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/27_text_11.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (852, 841), _c27)
except Exception:
    pass
layout["11"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/28_text_12.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (60, 961), _c28)
except Exception:
    pass
layout["12"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/29_text_13.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (192, 961), _c29)
except Exception:
    pass
layout["13"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/30_text_14.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (324, 961), _c30)
except Exception:
    pass
layout["14"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/31_text_15.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (456, 961), _c31)
except Exception:
    pass
layout["15"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/32_text_16.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (588, 961), _c32)
except Exception:
    pass
layout["16"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/33_text_17.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (720, 961), _c33)
except Exception:
    pass
layout["17"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/34_text_18.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (852, 961), _c34)
except Exception:
    pass
layout["18"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/35_text_19.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (60, 1081), _c35)
except Exception:
    pass
layout["19"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/36_text_20.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (192, 1081), _c36)
except Exception:
    pass
layout["20"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/37_text_21.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (324, 1081), _c37)
except Exception:
    pass
layout["21"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/38_text_24.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (720, 1081), _c38)
except Exception:
    pass
layout["24"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/39_text_25.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (852, 1081), _c39)
except Exception:
    pass
layout["25"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/40_text_30.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (588, 1201), _c40)
except Exception:
    pass
layout["30"] = [588, 1201, 720, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/41_text_31.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (720, 1201), _c41)
except Exception:
    pass
layout["31"] = [720, 1201, 852, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/42_text_End_Date.png
try:
    _c42 = get_crop(42, 252, 63)
    canvas.paste(_c42, (45, 1453), _c42)
except Exception:
    pass
layout["End_Date"] = [45, 1453, 297, 1516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (456, 721), _c43)
except Exception:
    pass
layout["1"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/44_clickable_2.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (588, 721), _c44)
except Exception:
    pass
layout["2"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/45_clickable_3.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (720, 721), _c45)
except Exception:
    pass
layout["3"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/46_clickable_4.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (852, 721), _c46)
except Exception:
    pass
layout["4"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/47_clickable_5.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (60, 841), _c47)
except Exception:
    pass
layout["5"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/48_clickable_6.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (192, 841), _c48)
except Exception:
    pass
layout["6"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/49_clickable_7.png
try:
    _c49 = get_crop(49, 132, 120)
    canvas.paste(_c49, (324, 841), _c49)
except Exception:
    pass
layout["7"] = [324, 841, 456, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/50_clickable_8.png
try:
    _c50 = get_crop(50, 132, 120)
    canvas.paste(_c50, (456, 841), _c50)
except Exception:
    pass
layout["8"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/51_clickable_9.png
try:
    _c51 = get_crop(51, 132, 120)
    canvas.paste(_c51, (588, 841), _c51)
except Exception:
    pass
layout["9"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_11_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-13/52_clickable_Choose_a_date.png
try:
    _c52 = get_crop(52, 638, 144)
    canvas.paste(_c52, (48, 1490), _c52)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]
