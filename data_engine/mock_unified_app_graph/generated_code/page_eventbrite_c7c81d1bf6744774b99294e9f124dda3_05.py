# page_id: page_eventbrite_c7c81d1bf6744774b99294e9f124dda3_05
# screenshot: 2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7.png
# step_index: 5/10
# task: Open Eventbrite. Search for "Fitness". Select the events in the location "Chicago". What is the price of the first event in listing?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the mobile page
# Available variables: canvas (PIL Image 1440x2960), draw (PIL ImageDraw), fonts available

W, H = canvas.size

# Base background (mostly white)
draw.rectangle((0, 0, W - 1, H - 1), fill=(255, 255, 255))

# Status bar (top strip)
status_h = 72
draw.rectangle((0, 0, W - 1, status_h), fill=(205, 205, 205))

# Thin line separating status bar from toolbar
draw.line((0, status_h, W - 1, status_h), fill=(190, 190, 190), width=1)

# Toolbar / header area (under status bar)
toolbar_top = status_h
toolbar_bottom = 180
draw.rectangle((0, toolbar_top, W - 1, toolbar_bottom), fill=(255, 255, 255))

# Subtle bottom divider under header area
header_div_y = 240
draw.line((48, header_div_y, W - 48, header_div_y), fill=(200, 196, 206), width=2)

# Horizontal rule under the "Find events in..." heading (light muted line)
heading_rule_y = 336
draw.line((48, heading_rule_y, W - 48, heading_rule_y), fill=(170, 165, 180), width=2)

# Light separator under the options row (keeps content visually grouped)
options_sep_y = 560
draw.line((48, options_sep_y, W - 48, options_sep_y), fill=(240, 240, 245), width=1)

# Browsing/location card background (rounded rectangle)
browse_card_top = 700
browse_card_bottom = 920
card_margin_left = 40
card_margin_right = W - 40
draw.rounded_rectangle((card_margin_left, browse_card_top, card_margin_right, browse_card_bottom),
                       radius=14, fill=(250, 250, 252), outline=None)

# Subtle shadow line under the browsing card to lift it slightly
draw.rectangle((card_margin_left, browse_card_bottom, card_margin_right, browse_card_bottom + 4),
               fill=(235, 235, 240))

# A faint large content band below (keeps the remainder subtle and clean)
content_band_top = browse_card_bottom + 24
draw.rectangle((0, content_band_top, W - 1, H - 1), fill=(255, 255, 255))

# Light left margin guide line (very subtle) to indicate content column
draw.line((48, toolbar_bottom + 40, 48, H - 40), fill=(247, 247, 249), width=1)

# Right-side subtle vertical guide (keeps balance)
draw.line((W - 48, toolbar_bottom + 40, W - 48, H - 40), fill=(247, 247, 249), width=1)

# Final thin footer divider near bottom of top content area for visual separation
draw.line((48, content_band_top, W - 48, content_band_top), fill=(245, 245, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/01_icon_icon_1.png
try:
    _c1 = get_crop(1, 60, 61)
    canvas.paste(_c1, (310, 4), _c1)
except Exception:
    pass
layout["icon_1"] = [310, 4, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/02_icon_7.09.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.09"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/03_icon_7.09.png
try:
    _c3 = get_crop(3, 58, 65)
    canvas.paste(_c3, (116, 1), _c3)
except Exception:
    pass
layout["7.09"] = [116, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/04_icon_7.09.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.09"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 49, 57)
    canvas.paste(_c5, (249, 6), _c5)
except Exception:
    pass
layout["icon_5"] = [249, 6, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 60, 65)
    canvas.paste(_c6, (1213, 0), _c6)
except Exception:
    pass
layout["icon_6"] = [1213, 0, 1273, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 42, 63)
    canvas.paste(_c7, (1271, 1), _c7)
except Exception:
    pass
layout["icon_7"] = [1271, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 46, 59)
    canvas.paste(_c8, (1322, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1322, 3, 1368, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 51, 65)
    canvas.paste(_c9, (383, 1), _c9)
except Exception:
    pass
layout["icon_9"] = [383, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/10_text_7.09.png
try:
    _c10 = get_crop(10, 91, 45)
    canvas.paste(_c10, (20, 15), _c10)
except Exception:
    pass
layout["7.09"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/14_text_Current_location.png
try:
    _c14 = get_crop(14, 415, 114)
    canvas.paste(_c14, (48, 465), _c14)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/15_text_Virtual_attendance.png
try:
    _c15 = get_crop(15, 452, 114)
    canvas.paste(_c15, (511, 465), _c15)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/16_text_Browsing_in.png
try:
    _c16 = get_crop(16, 228, 55)
    canvas.paste(_c16, (44, 742), _c16)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/17_text_New_York.png
try:
    _c17 = get_crop(17, 1440, 138)
    canvas.paste(_c17, (0, 816), _c17)
except Exception:
    pass
layout["New_York"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/c7c81d1bf6744774b99294e9f124dda3/step_05_2024_4_23_19_8_c7c81d1bf6744774b99294e9f124dda3-7/18_text_New_York.png
try:
    _c18 = get_crop(18, 182, 50)
    canvas.paste(_c18, (44, 905), _c18)
except Exception:
    pass
layout["New_York"] = [44, 905, 226, 955]
