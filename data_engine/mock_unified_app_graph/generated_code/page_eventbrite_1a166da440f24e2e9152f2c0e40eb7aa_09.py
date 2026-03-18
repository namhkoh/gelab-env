# page_id: page_eventbrite_1a166da440f24e2e9152f2c0e40eb7aa_09
# screenshot: 2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11.png
# step_index: 9/16
# task: Open Eventbrite. Check "Sports" category. Filter events happening next month. Add the first event to your wishlist.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structural elements for the UI page (background, status bar, dividers, section/card backgrounds)
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Ensure full background fill (dominant color - white)
draw.rectangle([(0, 0), canvas.size], fill="#FFFFFF")

# Status bar area at very top (light gray background to match screenshot status bar)
status_h = 72
draw.rectangle([(0, 0), (1440, status_h)], fill="#CFCFCF")

# Subtle bottom edge/shadow for status bar
draw.line([(0, status_h), (1440, status_h)], fill="#B8B8B8", width=1)

# Header divider (under the header/title area) - thin subtle divider in warm purple-gray
header_divider_y = 200
draw.line([(32, header_divider_y), (1408, header_divider_y)], fill="#F1EFFF", width=1)

# Calendar area subtle separator under month navigation (light gray)
# Place it roughly around where the calendar grid visually ends (approx y ~1200 in the screenshot)
calendar_sep_y = 1320
draw.line([(48, calendar_sep_y), (1392, calendar_sep_y)], fill="#F5F5F7", width=1)

# Large section card background for the calendar region (very subtle, almost white)
# We keep it subtle to avoid obscuring pasted content; rounded rect behind calendar content
cal_card_top = 160
cal_card_left = 32
cal_card_right = 1408
cal_card_bottom = 1550
draw.rounded_rectangle(
    [(cal_card_left, cal_card_top), (cal_card_right, cal_card_bottom)],
    radius=16,
    fill="#FFFFFF",
    outline=None
)

# End Date section card background (very light gray tint to hint a section)
end_card_top = 1480
end_card_left = 32
end_card_right = 1408
end_card_bottom = 2050
draw.rounded_rectangle(
    [(end_card_left, end_card_top), (end_card_right, end_card_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline=None
)

# Light separator line above the end date area (to separate from calendar)
draw.line([(48, end_card_top), (1392, end_card_top)], fill="#EFEFF2", width=1)

# Bottom "Apply date range" button background (rounded rectangle with border)
btn_left, btn_top = 48, 2768
btn_w, btn_h = 1344, 144
btn_right, btn_bottom = btn_left + btn_w, btn_top + btn_h

# Soft shadow behind the button
shadow_offset = 6
shadow_rect = [(btn_left, btn_top + shadow_offset), (btn_right, btn_bottom + shadow_offset)]
draw.rounded_rectangle(shadow_rect, radius=12, fill="#E8E8EA")

# Button background (white) with subtle border in muted purple/gray
draw.rounded_rectangle(
    [(btn_left, btn_top), (btn_right, btn_bottom)],
    radius=12,
    fill="#FFFFFF",
    outline="#BDB6C2",
    width=3
)

# Top thin divider above the button area to separate content from controls
draw.line([(32, btn_top - 12), (1408, btn_top - 12)], fill="#F0EEF4", width=1)

# Left navigation touch target highlight area background (behind the back arrow icon)
# Keep very subtle: small rounded rect on upper-left to mark the tappable region
nav_box = [(24, 88), (120, 176)]
draw.rounded_rectangle(nav_box, radius=10, fill="#FFFFFF")  # white base (keeps consistent look)
draw.rounded_rectangle(nav_box, radius=10, outline="#EFEFF2", width=1)

# Right top status cluster background (subtle rounded rect behind battery/signal icons)
status_cluster = [(1260, 12), (1436, 64)]
draw.rounded_rectangle(status_cluster, radius=10, fill="#CFCFCF", outline="#BFBFBF", width=1)

# Minor vertical separators for visual grouping (very subtle)
draw.line([(480, cal_card_top+12), (480, cal_card_bottom-12)], fill="#FFFFFF", width=1)
draw.line([(960, cal_card_top+12), (960, cal_card_bottom-12)], fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/00_icon_24.png
try:
    _c0 = get_crop(0, 132, 120)
    canvas.paste(_c0, (456, 1081), _c0)
except Exception:
    pass
layout["24"] = [456, 1081, 588, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/01_icon_Apply_date_range.png
try:
    _c1 = get_crop(1, 1344, 144)
    canvas.paste(_c1, (48, 2768), _c1)
except Exception:
    pass
layout["Apply_date_range"] = [48, 2768, 1392, 2912]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/02_icon_28.png
try:
    _c2 = get_crop(2, 132, 120)
    canvas.paste(_c2, (60, 1201), _c2)
except Exception:
    pass
layout["28"] = [60, 1201, 192, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/03_icon_29.png
try:
    _c3 = get_crop(3, 132, 120)
    canvas.paste(_c3, (192, 1201), _c3)
except Exception:
    pass
layout["29"] = [192, 1201, 324, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/04_icon_23.png
try:
    _c4 = get_crop(4, 132, 120)
    canvas.paste(_c4, (324, 1081), _c4)
except Exception:
    pass
layout["23"] = [324, 1081, 456, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/05_icon_25.png
try:
    _c5 = get_crop(5, 132, 120)
    canvas.paste(_c5, (588, 1081), _c5)
except Exception:
    pass
layout["25"] = [588, 1081, 720, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/06_icon_30.png
try:
    _c6 = get_crop(6, 132, 120)
    canvas.paste(_c6, (324, 1201), _c6)
except Exception:
    pass
layout["30"] = [324, 1201, 456, 1321]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/07_icon_27.png
try:
    _c7 = get_crop(7, 132, 120)
    canvas.paste(_c7, (852, 1081), _c7)
except Exception:
    pass
layout["27"] = [852, 1081, 984, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/08_icon_26.png
try:
    _c8 = get_crop(8, 132, 120)
    canvas.paste(_c8, (720, 1081), _c8)
except Exception:
    pass
layout["26"] = [720, 1081, 852, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/09_icon_5.31.png
try:
    _c9 = get_crop(9, 61, 65)
    canvas.paste(_c9, (180, 0), _c9)
except Exception:
    pass
layout["5.31"] = [180, 0, 241, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/10_icon_5.31.png
try:
    _c10 = get_crop(10, 64, 66)
    canvas.paste(_c10, (111, 1), _c10)
except Exception:
    pass
layout["5.31"] = [111, 1, 175, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/11_icon_icon_11.png
try:
    _c11 = get_crop(11, 63, 64)
    canvas.paste(_c11, (309, 2), _c11)
except Exception:
    pass
layout["icon_11"] = [309, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/12_icon_21.png
try:
    _c12 = get_crop(12, 132, 120)
    canvas.paste(_c12, (60, 1081), _c12)
except Exception:
    pass
layout["21"] = [60, 1081, 192, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/13_icon_22.png
try:
    _c13 = get_crop(13, 132, 120)
    canvas.paste(_c13, (192, 1081), _c13)
except Exception:
    pass
layout["22"] = [192, 1081, 324, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 53, 64)
    canvas.paste(_c14, (247, 2), _c14)
except Exception:
    pass
layout["icon_14"] = [247, 2, 300, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 57, 70)
    canvas.paste(_c15, (1316, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1316, 0, 1373, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/16_icon_18.png
try:
    _c16 = get_crop(16, 132, 120)
    canvas.paste(_c16, (588, 961), _c16)
except Exception:
    pass
layout["18"] = [588, 961, 720, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/17_icon_5.31.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (12, 72), _c17)
except Exception:
    pass
layout["5.31"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 91, 69)
    canvas.paste(_c18, (1211, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1211, 0, 1302, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/19_icon_11.png
try:
    _c19 = get_crop(19, 132, 120)
    canvas.paste(_c19, (588, 721), _c19)
except Exception:
    pass
layout["11"] = [588, 721, 720, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/20_icon_19.png
try:
    _c20 = get_crop(20, 132, 120)
    canvas.paste(_c20, (720, 961), _c20)
except Exception:
    pass
layout["19"] = [720, 961, 852, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/21_icon_April_2024.png
try:
    _c21 = get_crop(21, 126, 110)
    canvas.paste(_c21, (593, 611), _c21)
except Exception:
    pass
layout["April_2024"] = [593, 611, 719, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 49, 67)
    canvas.paste(_c22, (382, 1), _c22)
except Exception:
    pass
layout["icon_22"] = [382, 1, 431, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 41, 65)
    canvas.paste(_c23, (1274, 0), _c23)
except Exception:
    pass
layout["icon_23"] = [1274, 0, 1315, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/24_icon_5.31.png
try:
    _c24 = get_crop(24, 93, 65)
    canvas.paste(_c24, (15, 1), _c24)
except Exception:
    pass
layout["5.31"] = [15, 1, 108, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/25_icon_Next_month.png
try:
    _c25 = get_crop(25, 144, 144)
    canvas.paste(_c25, (846, 457), _c25)
except Exception:
    pass
layout["Next_month"] = [846, 457, 990, 601]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/26_icon_12.png
try:
    _c26 = get_crop(26, 132, 120)
    canvas.paste(_c26, (720, 721), _c26)
except Exception:
    pass
layout["12"] = [720, 721, 852, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/27_icon_12.png
try:
    _c27 = get_crop(27, 104, 107)
    canvas.paste(_c27, (733, 614), _c27)
except Exception:
    pass
layout["12"] = [733, 614, 837, 721]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/28_icon_Choose_a_date.png
try:
    _c28 = get_crop(28, 638, 144)
    canvas.paste(_c28, (48, 1490), _c28)
except Exception:
    pass
layout["Choose_a_date"] = [48, 1490, 686, 1634]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 104, 100)
    canvas.paste(_c29, (71, 618), _c29)
except Exception:
    pass
layout["icon_29"] = [71, 618, 175, 718]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/30_icon_What_date.png
try:
    _c30 = get_crop(30, 322, 71)
    canvas.paste(_c30, (558, 113), _c30)
except Exception:
    pass
layout["What_date?"] = [558, 113, 880, 184]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/31_icon_16.png
try:
    _c31 = get_crop(31, 132, 120)
    canvas.paste(_c31, (324, 961), _c31)
except Exception:
    pass
layout["16"] = [324, 961, 456, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/32_icon_10.png
try:
    _c32 = get_crop(32, 132, 120)
    canvas.paste(_c32, (324, 721), _c32)
except Exception:
    pass
layout["10"] = [324, 721, 456, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/33_text_Start_Date.png
try:
    _c33 = get_crop(33, 589, 114)
    canvas.paste(_c33, (48, 313), _c33)
except Exception:
    pass
layout["Start_Date"] = [48, 313, 637, 427]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/34_text_April_2024.png
try:
    _c34 = get_crop(34, 203, 54)
    canvas.paste(_c34, (420, 504), _c34)
except Exception:
    pass
layout["April_2024"] = [420, 504, 623, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/35_text_10.png
try:
    _c35 = get_crop(35, 132, 120)
    canvas.paste(_c35, (456, 841), _c35)
except Exception:
    pass
layout["10"] = [456, 841, 588, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/36_text_11.png
try:
    _c36 = get_crop(36, 132, 120)
    canvas.paste(_c36, (588, 841), _c36)
except Exception:
    pass
layout["11"] = [588, 841, 720, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/37_text_12.png
try:
    _c37 = get_crop(37, 132, 120)
    canvas.paste(_c37, (720, 841), _c37)
except Exception:
    pass
layout["12"] = [720, 841, 852, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/38_text_13.png
try:
    _c38 = get_crop(38, 132, 120)
    canvas.paste(_c38, (852, 841), _c38)
except Exception:
    pass
layout["13"] = [852, 841, 984, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/39_text_14.png
try:
    _c39 = get_crop(39, 132, 120)
    canvas.paste(_c39, (60, 961), _c39)
except Exception:
    pass
layout["14"] = [60, 961, 192, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/40_text_15.png
try:
    _c40 = get_crop(40, 132, 120)
    canvas.paste(_c40, (192, 961), _c40)
except Exception:
    pass
layout["15"] = [192, 961, 324, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/41_text_17.png
try:
    _c41 = get_crop(41, 132, 120)
    canvas.paste(_c41, (456, 961), _c41)
except Exception:
    pass
layout["17"] = [456, 961, 588, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/42_text_20.png
try:
    _c42 = get_crop(42, 132, 120)
    canvas.paste(_c42, (852, 961), _c42)
except Exception:
    pass
layout["20"] = [852, 961, 984, 1081]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/43_clickable_1.png
try:
    _c43 = get_crop(43, 132, 120)
    canvas.paste(_c43, (192, 721), _c43)
except Exception:
    pass
layout["1"] = [192, 721, 324, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/44_clickable_3.png
try:
    _c44 = get_crop(44, 132, 120)
    canvas.paste(_c44, (456, 721), _c44)
except Exception:
    pass
layout["3"] = [456, 721, 588, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/45_clickable_6.png
try:
    _c45 = get_crop(45, 132, 120)
    canvas.paste(_c45, (852, 721), _c45)
except Exception:
    pass
layout["6"] = [852, 721, 984, 841]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/46_clickable_7.png
try:
    _c46 = get_crop(46, 132, 120)
    canvas.paste(_c46, (60, 841), _c46)
except Exception:
    pass
layout["7"] = [60, 841, 192, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/47_clickable_8.png
try:
    _c47 = get_crop(47, 132, 120)
    canvas.paste(_c47, (192, 841), _c47)
except Exception:
    pass
layout["8"] = [192, 841, 324, 961]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1a166da440f24e2e9152f2c0e40eb7aa/step_09_2024_4_24_17_29_1a166da440f24e2e9152f2c0e40eb7aa-11/48_clickable_9.png
try:
    _c48 = get_crop(48, 132, 120)
    canvas.paste(_c48, (324, 841), _c48)
except Exception:
    pass
layout["9"] = [324, 841, 456, 961]
