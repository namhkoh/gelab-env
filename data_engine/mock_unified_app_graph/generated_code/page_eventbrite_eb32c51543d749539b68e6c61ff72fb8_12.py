# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_12
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14.png
# step_index: 12/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structure for the UI mockup using provided canvas and draw objects.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), fonts: font_sm,font_md,font_lg,font_xl

# Full white background (canvas starts white but ensure fill)
draw.rectangle([(0, 0), (1440, 2960)], fill="#FFFFFF")

# Status bar area at top (~72px) - light gray to match screenshot
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#D0D0D0")

# Thin darker top hairline for status bar (subtle)
draw.line([(0, status_h-1), (1440, status_h-1)], fill="#B8B8B8", width=1)

# Header area (toolbar) below status bar - keep white but give subtle divider
header_top = status_h
header_bottom = 180
draw.rectangle([(0, header_top), (1440, header_bottom)], fill="#FFFFFF")
# Subtle bottom divider under header
draw.line([(24, header_bottom), (1440-24, header_bottom)], fill="#EFEFF2", width=1)

# Main calendar/card background (rounded) behind calendar content
card_left = 40
card_right = 1400
card_top = 200
card_bottom = 1320
corner = 24
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                       radius=corner, fill="#FBF9FF", outline="#F0EDF8", width=1)

# Light grid area background for month selector row (inside card) - subtle band behind month controls
month_band_top = card_top + 36
month_band_bottom = month_band_top + 88
draw.rectangle([(card_left + 28, month_band_top), (card_right - 28, month_band_bottom)],
               fill="#FFFFFF", outline=None)

# Separator line between calendar card and the "End Date" section
sep_y = card_bottom + 36
draw.line([(24, sep_y), (1440-24, sep_y)], fill="#F0EDF8", width=1)

# End-date section card (large white area with faint border)
end_left = 24
end_right = 1440 - 24
end_top = sep_y + 24
end_bottom = 2620
draw.rounded_rectangle([(end_left, end_top), (end_right, end_bottom)],
                       radius=18, fill="#FFFFFF", outline="#F3F1F6", width=1)

# Subtle shadow line above the bottom action bar area (do not draw the actual button)
action_bar_top = 2760
draw.line([(24, action_bar_top-32), (1440-24, action_bar_top-32)], fill="#EFEFF2", width=1)
# Slight darker divider at the very top of the area where the bottom action will appear
draw.line([(24, action_bar_top-4), (1440-24, action_bar_top-4)], fill="#DFDBE6", width=1)

# Add an inner faint guide box for where the calendar grid sits (non-intrusive background only)
grid_left = card_left + 40
grid_right = card_right - 40
grid_top = month_band_bottom + 24
grid_bottom = card_bottom - 40
draw.rectangle([(grid_left, grid_top), (grid_right, grid_bottom)], fill="#FFFFFF", outline=None)

# Additional subtle vertical rhythm lines to suggest calendar columns (very faint)
col_width = (grid_right - grid_left) / 7.0
for i in range(1, 7):
    x = grid_left + int(i * col_width)
    draw.line([(x, grid_top + 18), (x, grid_bottom - 18)], fill="#FBF9FF", width=1)

# Lightweight horizontal separators to suggest rows (do not overlap exact day number areas strongly)
row_height = (grid_bottom - grid_top) / 6.0
for r in range(1, 6):
    y = int(grid_top + r * row_height)
    draw.line([(grid_left + 12, y), (grid_right - 12, y)], fill="#FBF9FF", width=1)

# Decorative left edge highlight on card (soft)
draw.rectangle([(card_left, card_top), (card_left + 6, card_bottom)], fill="#F6F4FB")

# Decorative right edge highlight on end-date card
draw.rectangle([(end_right - 6, end_top), (end_right, end_bottom)], fill="#F6F4FB")

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/00_icon_Apply_date_range.png
try:
    _c0 = get_crop(0, 1344, 144)
    canvas.paste(_c0, (48, 2768), _c0)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 52, 71)
    canvas.paste(_c1, (1153, 0), _c1)
except Exception:
    pass
layout["icon_1"] = [1153, 0, 1205, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/02_icon_7.48.png
try:
    _c2 = get_crop(2, 62, 66)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["7.48"] = [179, 1, 241, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/03_icon_28.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (324, 1201), _c3)
except Exception:
    pass
layout["28"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 66, 65)
    canvas.paste(_c4, (308, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [308, 2, 374, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/05_icon_7.48.png
try:
    _c5 = get_crop(5, 64, 68)
    canvas.paste(_c5, (112, 0), _c5)
except Exception:
    pass
layout["7.48"] = [112, 0, 176, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 102, 70)
    canvas.paste(_c6, (1210, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1210, 0, 1312, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/07_icon_26.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (60, 1201), _c7)
except Exception:
    pass
layout["26"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/08_icon_27.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (192, 1201), _c8)
except Exception:
    pass
layout["27"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 53, 65)
    canvas.paste(_c9, (247, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [247, 1, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 71)
    canvas.paste(_c10, (1318, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1318, 0, 1371, 71]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/11_icon_7.48.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (12, 72), _c11)
except Exception:
    pass
layout["7.48"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/12_icon_May.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (54, 457), _c12)
except Exception:
    pass
layout["May"] = [54, 457, 198, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 92, 105)
    canvas.paste(_c13, (76, 617), _c13)
except Exception:
    pass
layout["icon_13"] = [76, 617, 168, 722]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/14_icon_29.png
try:
    _c14 = get_crop(14, 132, 120)
    canvas.paste(_c14, (456, 1201), _c14)
except Exception:
    pass
layout["29"] = [456, 1201, 588, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 50, 68)
    canvas.paste(_c15, (382, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [382, 1, 432, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/16_icon_7.48.png
try:
    _c16 = get_crop(16, 93, 65)
    canvas.paste(_c16, (15, 1), _c16)
except Exception:
    pass
layout["7.48"] = [15, 1, 108, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/17_icon_15.png
try:
    _c17 = get_crop(17, 132, 120)
    canvas.paste(_c17, (456, 841), _c17)
except Exception:
    pass
layout["15"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/18_icon_What_date.png
try:
    _c18 = get_crop(18, 321, 71)
    canvas.paste(_c18, (558, 113), _c18)
except Exception:
    pass
layout["What_date?"] = [558, 113, 879, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/19_icon_May.png
try:
    _c19 = get_crop(19, 114, 93)
    canvas.paste(_c19, (462, 619), _c19)
except Exception:
    pass
layout["May"] = [462, 619, 576, 712]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/20_icon_Next_month.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (846, 457), _c20)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/21_icon_Choose_a_date.png
try:
    _c21 = get_crop(21, 638, 144)
    canvas.paste(_c21, (48, 1490), _c21)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/22_text_Start_Date.png
try:
    _c22 = get_crop(22, 591, 114)
    canvas.paste(_c22, (48, 313), _c22)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 639, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/23_text_10.png
try:
    _c23 = get_crop(23, 132, 120)
    canvas.paste(_c23, (720, 841), _c23)
except Exception:
    pass
layout["10"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/24_text_11.png
try:
    _c24 = get_crop(24, 132, 120)
    canvas.paste(_c24, (852, 841), _c24)
except Exception:
    pass
layout["11"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/25_text_12.png
try:
    _c25 = get_crop(25, 132, 120)
    canvas.paste(_c25, (60, 961), _c25)
except Exception:
    pass
layout["12"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/26_text_13.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (192, 961), _c26)
except Exception:
    pass
layout["13"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/27_text_14.png
try:
    _c27 = get_crop(27, 132, 120)
    canvas.paste(_c27, (324, 961), _c27)
except Exception:
    pass
layout["14"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/28_text_15.png
try:
    _c28 = get_crop(28, 132, 120)
    canvas.paste(_c28, (456, 961), _c28)
except Exception:
    pass
layout["15"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/29_text_16.png
try:
    _c29 = get_crop(29, 132, 120)
    canvas.paste(_c29, (588, 961), _c29)
except Exception:
    pass
layout["16"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/30_text_17.png
try:
    _c30 = get_crop(30, 132, 120)
    canvas.paste(_c30, (720, 961), _c30)
except Exception:
    pass
layout["17"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/31_text_18.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (852, 961), _c31)
except Exception:
    pass
layout["18"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/32_text_19.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (60, 1081), _c32)
except Exception:
    pass
layout["19"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/33_text_20.png
try:
    _c33 = get_crop(33, 132, 120)
    canvas.paste(_c33, (192, 1081), _c33)
except Exception:
    pass
layout["20"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/34_text_21.png
try:
    _c34 = get_crop(34, 132, 120)
    canvas.paste(_c34, (324, 1081), _c34)
except Exception:
    pass
layout["21"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/35_text_22.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 1081), _c35)
except Exception:
    pass
layout["22"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/36_text_23.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 1081), _c36)
except Exception:
    pass
layout["23"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/37_text_24.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 1081), _c37)
except Exception:
    pass
layout["24"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/38_text_25.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 1081), _c38)
except Exception:
    pass
layout["25"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/39_text_30.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (588, 1201), _c39)
except Exception:
    pass
layout["30"] = [588, 1201, 720, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/40_text_31.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (720, 1201), _c40)
except Exception:
    pass
layout["31"] = [720, 1201, 852, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/41_clickable_1.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 721), _c41)
except Exception:
    pass
layout["1"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/42_clickable_2.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (588, 721), _c42)
except Exception:
    pass
layout["2"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/43_clickable_3.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (720, 721), _c43)
except Exception:
    pass
layout["3"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/44_clickable_4.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (852, 721), _c44)
except Exception:
    pass
layout["4"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/45_clickable_5.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (60, 841), _c45)
except Exception:
    pass
layout["5"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/46_clickable_6.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (192, 841), _c46)
except Exception:
    pass
layout["6"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/47_clickable_7.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (324, 841), _c47)
except Exception:
    pass
layout["7"] = [324, 841, 456, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_12_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-14/48_clickable_9.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (588, 841), _c48)
except Exception:
    pass
layout["9"] = [588, 841, 720, 961]
