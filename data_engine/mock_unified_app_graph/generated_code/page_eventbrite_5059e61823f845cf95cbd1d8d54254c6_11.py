# page_id: page_eventbrite_5059e61823f845cf95cbd1d8d54254c6_11
# screenshot: 2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13.png
# step_index: 11/19
# task: Open Eventbrite. Look for 'Education' in Los Angeles happening on May 4. Filter to show only free events. How many events are posted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw UI background and structure for a 1440x2960 canvas
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_color = (255, 255, 255)         # main white background
status_bar_color = (190, 190, 190) # muted grey for status bar
header_divider = (236, 233, 239)   # subtle divider under header
shadow_color = (224, 219, 224)     # faint shadow line
separator_color = (242, 240, 243)  # very light separators

# Fill full background (reinforce white)
draw.rectangle((0, 0, w, h), fill=bg_color)

# Status bar (top area) - approximate height 88px to match screenshot feel
status_bar_height = 88
draw.rectangle((0, 0, w, status_bar_height), fill=status_bar_color)

# Header / toolbar area - leave it visually consistent with background but add subtle bottom divider/shadow
# Header area extends roughly from status_bar_height to below the title text (we'll mark a divider around y=220)
header_bottom_y = 220
# Slightly lighter area for header (same as background to keep text/icons pasted on top readable)
draw.rectangle((0, status_bar_height, w, header_bottom_y), fill=bg_color)

# Divider / shadow under header
draw.rectangle((0, header_bottom_y - 2, w, header_bottom_y), fill=header_divider)
draw.line((0, header_bottom_y, w, header_bottom_y), fill=shadow_color, width=1)

# Section separators between the large list items.
# Use the detected text positions to place subtle separators just above each list item block.
separators_y = [
    378,  # just below title block (When do you want to go out?)
    558,  # below "Today"
    738,  # below "Tomorrow"
    918,  # below "This Week"
    1098, # below "This Weekend"
    1278, # below "Choose a date..."
]
left_margin = 48
right_margin = w - 48

for y in separators_y:
    # draw a very light 1px line across the content width
    draw.line((left_margin, y, right_margin, y), fill=separator_color, width=1)

# Subtle left alignment guide/background strip (very faint) for the content column
# This creates a soft visual column but remains unobtrusive (won't overlap pasted icons/text visibly)
draw.rectangle((left_margin - 12, header_bottom_y + 12, left_margin - 8, h), fill=(250,250,250))

# Add a faint large-area card/background behind the first group (to hint grouping without drawing text/icons).
# Place it from just under the header to slightly below the last visible option area.
card_top = header_bottom_y + 4
card_bottom = 1320
card_margin = 24
card_bbox = (card_margin, card_top, w - card_margin, card_bottom)
# Large rounded rectangle with very small corner radius
try:
    draw.rounded_rectangle(card_bbox, radius=12, fill=bg_color, outline=(245,245,246), width=1)
except Exception:
    # Fallback in case rounded_rectangle is not available
    draw.rectangle(card_bbox, fill=bg_color, outline=(245,245,246), width=1)

# Soft drop shadow under the card to separate it from header area
shadow_y = card_bottom + 2
draw.line((card_margin + 2, shadow_y, w - card_margin - 2, shadow_y), fill=(248,247,249), width=2)

# Ensure top-left notch area of status bar has a slight inner darker line to mimic device chrome
draw.line((0, status_bar_height - 1, w, status_bar_height - 1), fill=(180,180,180), width=1)

# End of background/structure drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/00_icon_7.35.png
try:
    _c0 = get_crop(0, 58, 61)
    canvas.paste(_c0, (181, 3), _c0)
except Exception:
    pass
layout["7.35"] = [181, 3, 239, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/01_icon_7.35.png
try:
    _c1 = get_crop(1, 56, 63)
    canvas.paste(_c1, (116, 3), _c1)
except Exception:
    pass
layout["7.35"] = [116, 3, 172, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 64, 61)
    canvas.paste(_c2, (308, 3), _c2)
except Exception:
    pass
layout["icon_2"] = [308, 3, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/03_icon_7.35.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (12, 72), _c3)
except Exception:
    pass
layout["7.35"] = [12, 72, 156, 216]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/04_icon_Anytime.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 234), _c4)
except Exception:
    pass
layout["Anytime"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 99, 62)
    canvas.paste(_c5, (1215, 2), _c5)
except Exception:
    pass
layout["icon_5"] = [1215, 2, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 48, 63)
    canvas.paste(_c6, (1154, 4), _c6)
except Exception:
    pass
layout["icon_6"] = [1154, 4, 1202, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 51, 58)
    canvas.paste(_c7, (248, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [248, 5, 299, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 60)
    canvas.paste(_c8, (1325, 4), _c8)
except Exception:
    pass
layout["icon_8"] = [1325, 4, 1371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/09_icon_7.35.png
try:
    _c9 = get_crop(9, 92, 62)
    canvas.paste(_c9, (16, 3), _c9)
except Exception:
    pass
layout["7.35"] = [16, 3, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 123, 128)
    canvas.paste(_c10, (1291, 247), _c10)
except Exception:
    pass
layout["icon_10"] = [1291, 247, 1414, 375]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/11_icon_Tomorrow.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 594), _c11)
except Exception:
    pass
layout["Tomorrow"] = [48, 594, 1392, 738]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/12_text_When_do_you_want_to_go_out.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 234), _c12)
except Exception:
    pass
layout["When_do_you_want_to_go_ou"] = [48, 234, 1392, 378]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/13_text_Today.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 414), _c13)
except Exception:
    pass
layout["Today"] = [48, 414, 1392, 558]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/14_text_This_Week.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 774), _c14)
except Exception:
    pass
layout["This_Week"] = [48, 774, 1392, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/15_text_This_Weekend.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 954), _c15)
except Exception:
    pass
layout["This_Weekend"] = [48, 954, 1392, 1098]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/5059e61823f845cf95cbd1d8d54254c6/step_11_2024_4_23_19_33_5059e61823f845cf95cbd1d8d54254c6-13/16_text_Choose_a_date-.png
try:
    _c16 = get_crop(16, 1344, 144)
    canvas.paste(_c16, (48, 1134), _c16)
except Exception:
    pass
layout["Choose_a_date-"] = [48, 1134, 1392, 1278]
