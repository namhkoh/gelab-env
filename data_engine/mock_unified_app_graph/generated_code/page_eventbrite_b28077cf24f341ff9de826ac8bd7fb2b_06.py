# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_06
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8.png
# step_index: 6/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (PIL Image and ImageDraw). Fonts: font_sm, font_md, font_lg, font_xl
# Draw page background and structural elements only.

# Full background (slightly warm white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar (top subtle grey background)
status_h = 84
draw.rectangle([(0, 0), (1440, status_h)], fill=(150, 150, 150))

# Top toolbar area below status bar (clean white with a subtle bottom divider shadow)
toolbar_top = status_h
toolbar_h = 156
draw.rectangle([(0, toolbar_top), (1440, toolbar_h)], fill=(255, 255, 255))
# subtle shadow line under toolbar
draw.line([(0, toolbar_h), (1440, toolbar_h)], fill=(230, 230, 230), width=2)

# Large horizontal accent divider under the "Find events in..." area
# Position chosen to sit just below the heading region (do not draw text)
accent_left = 48
accent_right = 1392
accent_y = 354
draw.line([(accent_left, accent_y), (accent_right, accent_y)], fill=(63, 81, 181), width=4)

# Faint rounded container behind the "Nearby" row (subtle pale background)
nearby_card_top = 300
nearby_card_bottom = 420
draw.rounded_rectangle([(24, nearby_card_top), (1416, nearby_card_bottom)],
                       radius=12, fill=(250, 252, 255), outline=None)

# Subtle separator between the "Nearby" area and the next section
sep_y = 428
draw.line([(36, sep_y), (1404, sep_y)], fill=(240, 240, 240), width=1)

# Large section background for the "Browsing in / Online events" block
browse_top = 720
browse_bottom = 920
draw.rounded_rectangle([(24, browse_top), (1416, browse_bottom)],
                       radius=14, fill=(249, 246, 255), outline=None)

# Subtle right-side fade/band to suggest the check/action area (keeps icons separate)
band_x1 = 1240
band_x2 = 1440
band_top = browse_top + 40
band_bottom = browse_bottom - 40
draw.rectangle([(band_x1, band_top), (band_x2, band_bottom)], fill=(255, 255, 255, 0))

# Light horizontal separators further down to structure empty content area
y_positions = [980, 1250, 1600]
for y in y_positions:
    draw.line([(36, y), (1404, y)], fill=(245, 245, 245), width=1)

# Very subtle left margin guide/background for content groups (visual structure only)
draw.rectangle([(24, toolbar_h + 28), (36, 280)], fill=(250, 250, 250))
draw.rectangle([(24, browse_bottom + 28), (36, browse_bottom + 220)], fill=(250, 250, 250))

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/00_icon_4.44.png
try:
    _c0 = get_crop(0, 168, 168)
    canvas.paste(_c0, (0, 72), _c0)
except Exception:
    pass
layout["4.44"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/01_icon_4.44.png
try:
    _c1 = get_crop(1, 60, 64)
    canvas.paste(_c1, (180, 1), _c1)
except Exception:
    pass
layout["4.44"] = [180, 1, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/02_icon_4.44.png
try:
    _c2 = get_crop(2, 61, 65)
    canvas.paste(_c2, (113, 1), _c2)
except Exception:
    pass
layout["4.44"] = [113, 1, 174, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 62, 61)
    canvas.paste(_c3, (309, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [309, 3, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 50, 58)
    canvas.paste(_c4, (248, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [248, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 103, 108)
    canvas.paste(_c5, (1291, 836), _c5)
except Exception:
    pass
layout["icon_5"] = [1291, 836, 1394, 944]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/06_icon_4.44.png
try:
    _c6 = get_crop(6, 93, 62)
    canvas.paste(_c6, (15, 2), _c6)
except Exception:
    pass
layout["4.44"] = [15, 2, 108, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 47, 57)
    canvas.paste(_c7, (1322, 4), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 4, 1369, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 86, 61)
    canvas.paste(_c8, (1213, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1213, 1, 1299, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 42, 59)
    canvas.paste(_c9, (1272, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1272, 3, 1314, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 435, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/11_text_Find_events_in..png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Find_events_in._"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/13_text_Current_location.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/14_text_Browsing_in.png
try:
    _c14 = get_crop(14, 228, 55)
    canvas.paste(_c14, (44, 742), _c14)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/15_text_Online_events.png
try:
    _c15 = get_crop(15, 1440, 138)
    canvas.paste(_c15, (0, 816), _c15)
except Exception:
    pass
layout["Online_events"] = [0, 816, 1440, 954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_06_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-8/16_text_Virtual_attendance.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Virtual_attendance"] = [0, 816, 1440, 954]
