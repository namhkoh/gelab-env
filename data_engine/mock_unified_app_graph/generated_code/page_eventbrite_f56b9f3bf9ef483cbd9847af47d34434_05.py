# page_id: page_eventbrite_f56b9f3bf9ef483cbd9847af47d34434_05
# screenshot: 2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7.png
# step_index: 5/8
# task: Open Eventbrite. Look up "Gardening" events. Filter by events happening this week. Select the first event from the results. Follow the organizer and where is the event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structured background and containers for the mobile UI
# Assumes provided variables: canvas (PIL Image 1440x2960 RGB) and draw (ImageDraw.Draw),
# and font_sm, font_md, font_lg, font_xl (not used here).

# Full background
draw.rectangle([0, 0, 1440, 2960], fill="#FFFFFF")

# Status bar (top area with battery/time/icons) ~ 50-70px high
status_h = 64
draw.rectangle([0, 0, 1440, status_h], fill="#CFCFCF")

# Subtle top shadow under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#BFBFBF", width=1)

# Header / toolbar area (where the title sits) ~ from status_h to ~160px
header_top = status_h
header_bottom = 160
draw.rectangle([0, header_top, 1440, header_bottom], fill="#FFFFFF")

# Header bottom divider (subtle)
draw.line([(24, header_bottom), (1440 - 24, header_bottom)], fill="#E9E9E9", width=1)

# Decorative thin top border for header (very subtle)
draw.line([(24, header_top), (1440 - 24, header_top)], fill="#F3F3F3", width=1)

# Prepare rounded cards (backgrounds) behind each list item group
# Do NOT draw any text or icons - only the card backgrounds/outlines.
item_positions = [234, 414, 594, 774, 954, 1134]  # y positions from detected elements
card_left = 24
card_right = 1440 - 24
card_width = card_right - card_left
card_radius = 16

for y in item_positions:
    top = y - 8
    bottom = y + 144 - 8  # keep the same height as the detected text blocks
    # Slightly off-white fill to distinguish rows subtly; outline is light gray.
    draw.rounded_rectangle([card_left, top, card_right, bottom],
                           radius=card_radius,
                           fill="#FFFFFF",
                           outline="#F0F0F0",
                           width=1)

    # Add a subtle divider line at the bottom edge of the card
    draw.line([(card_left + 8, bottom), (card_right - 8, bottom)],
              fill="#F2F2F2", width=1)

# A subtle vertical padding guide (decorative, very light) on the left
draw.line([(48, header_bottom + 8), (48, 2960 - 24)], fill="#FBFBFB", width=1)

# Bottom area: gentle fade / large white area (no content) - draw a faint horizontal anchor
draw.line([(24, 1400), (1440 - 24, 1400)], fill="#FAFAFA", width=1)

# End of background/structure drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 63, 60)
    canvas.paste(_c0, (309, 4), _c0)
except Exception:
    pass
layout["icon_0"] = [309, 4, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/01_icon_5.09.png
try:
    _c1 = get_crop(1, 60, 62)
    canvas.paste(_c1, (180, 2), _c1)
except Exception:
    pass
layout["5.09"] = [180, 2, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/02_icon_5.09.png
try:
    _c2 = get_crop(2, 144, 144)
    canvas.paste(_c2, (12, 72), _c2)
except Exception:
    pass
layout["5.09"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/03_icon_Anytime.png
try:
    _c3 = get_crop(3, 1344, 144)
    canvas.paste(_c3, (48, 234), _c3)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/04_icon_5.09.png
try:
    _c4 = get_crop(4, 57, 64)
    canvas.paste(_c4, (116, 2), _c4)
except Exception:
    pass
layout["5.09"] = [116, 2, 173, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 51, 58)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 100, 60)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1312, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 44, 60)
    canvas.paste(_c7, (1326, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1326, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 123, 129)
    canvas.paste(_c8, (1291, 246), _c8)
except Exception:
    pass
layout["icon_8"] = [1291, 246, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/09_icon_Tomorrow.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 594), _c9)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/10_text_5.09.png
try:
    _c10 = get_crop(10, 91, 45)
    canvas.paste(_c10, (20, 15), _c10)
except Exception:
    pass
layout["5.09"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/11_text_When_do_you_want_to_go_out.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 234), _c11)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/12_text_Today.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 414), _c12)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/13_text_This_Week.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 774), _c13)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/14_text_This_Weekend.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 954), _c14)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f56b9f3bf9ef483cbd9847af47d34434/step_05_2024_4_24_17_3_f56b9f3bf9ef483cbd9847af47d34434-7/15_text_Choose_a_date-.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1134), _c15)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
