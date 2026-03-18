# page_id: page_eventbrite_3c2c0d71896b45acb211a472de4b4c9e_03
# screenshot: 2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5.png
# step_index: 3/15
# task: Open Eventbrite. Search free Health event in Los Angeles. Select the first one that is not promoted. Follow the organizer. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# background and structural UI drawing for 1440x2960 canvas
# available: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# colors (matching the screenshot's subtle neutral + purple-gray accents)
BG = "#ffffff"
STATUS_BG = "#f4f6f8"        # light status bar background
DIVIDER = "#e0dfe6"          # thin divider (purple-gray)
SECTION_BG = "#fbfbfd"       # very light section card background
SUBTLE_LINE = "#f0eef3"      # faint separators

w, h = canvas.size

# Fill background (canvas starts white but ensure color)
draw.rectangle([(0, 0), (w, h)], fill=BG)

# Status bar area at top (~0..92)
status_h = 92
draw.rectangle([(0, 0), (w, status_h)], fill=STATUS_BG)
# subtle bottom line under status bar
draw.line([(0, status_h), (w, status_h)], fill=DIVIDER, width=1)

# Header area: large whitespace where page title sits.
# draw a thin divider under the header/title (approx y ~332)
header_div_y = 332
draw.line([(48, header_div_y), (w-48, header_div_y)], fill=DIVIDER, width=2)

# Section card: options row (Nearby / Online events) background card
opts_top = 360
opts_bottom = 560
opts_pad_x = 48
draw.rounded_rectangle(
    [(opts_pad_x, opts_top), (w - opts_pad_x, opts_bottom)],
    radius=18,
    fill=SECTION_BG,
    outline=None
)
# very subtle top and bottom separators inside the options card
draw.line([(opts_pad_x+12, opts_top+1), (w-opts_pad_x-12, opts_top+1)], fill=SUBTLE_LINE, width=1)
draw.line([(opts_pad_x+12, opts_bottom-1), (w-opts_pad_x-12, opts_bottom-1)], fill=SUBTLE_LINE, width=1)

# Separator between options and browsing section
sep_y = opts_bottom + 28
draw.line([(48, sep_y), (w-48, sep_y)], fill=SUBTLE_LINE, width=1)

# Browsing section card/background
browse_top = 720
browse_bottom = 980
draw.rounded_rectangle(
    [(32, browse_top), (w-32, browse_bottom)],
    radius=20,
    fill=SECTION_BG,
    outline=None
)
# subtle divider inside browsing card (to hint list separation)
inner_div_y = browse_top + 120
draw.line([(48, inner_div_y), (w-48, inner_div_y)], fill=DIVIDER, width=1)

# Row for current location ("New York") - background row to contain selection
row_top = browse_top + 80
row_bottom = row_top + 140
row_left = 48
row_right = w - 48
draw.rectangle([(row_left, row_top), (row_right, row_bottom)], fill=BG)  # keep row white but ensure separation
# faint bottom border for the item row
draw.line([(row_left, row_bottom), (row_right, row_bottom)], fill=SUBTLE_LINE, width=1)

# Decorative right-side circular placeholder background for selection (light)
# Note: do not draw any actual icons; this is only a subtle structural background behind where an icon may appear.
# Place it where detected check icon will be pasted (but keep as very faint ring so not duplicating the icon).
circle_cx = 1310
circle_cy = row_top + 40
circle_r = 56
draw.ellipse(
    [(circle_cx - circle_r, circle_cy - circle_r), (circle_cx + circle_r, circle_cy + circle_r)],
    fill="#fafafc"
)
# very faint ring
draw.ellipse(
    [(circle_cx - circle_r, circle_cy - circle_r), (circle_cx + circle_r, circle_cy + circle_r)],
    outline=SUBTLE_LINE,
    width=1
)

# Final subtle bottom divider to finish the structural layout
draw.line([(0, h-1), (w, h-1)], fill=SUBTLE_LINE, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/01_icon_9.41.png
try:
    _c1 = get_crop(1, 58, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["9.41"] = [180, 1, 238, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/02_icon_9.41.png
try:
    _c2 = get_crop(2, 53, 63)
    canvas.paste(_c2, (115, 2), _c2)
except Exception:
    pass
layout["9.41"] = [115, 2, 168, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/03_icon_9.41.png
try:
    _c3 = get_crop(3, 168, 168)
    canvas.paste(_c3, (0, 72), _c3)
except Exception:
    pass
layout["9.41"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 53, 62)
    canvas.paste(_c4, (315, 2), _c4)
except Exception:
    pass
layout["icon_4"] = [315, 2, 368, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 54, 62)
    canvas.paste(_c5, (247, 1), _c5)
except Exception:
    pass
layout["icon_5"] = [247, 1, 301, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 61, 63)
    canvas.paste(_c6, (1213, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 1, 1274, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 52, 62)
    canvas.paste(_c8, (1261, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1261, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 48, 64)
    canvas.paste(_c9, (383, 0), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 0, 431, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/10_text_9.41.png
try:
    _c10 = get_crop(10, 93, 50)
    canvas.paste(_c10, (18, 12), _c10)
except Exception:
    pass
layout["9.41"] = [18, 12, 111, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3c2c0d71896b45acb211a472de4b4c9e/step_03_2024_3_20_17_40_3c2c0d71896b45acb211a472de4b4c9e-5/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
