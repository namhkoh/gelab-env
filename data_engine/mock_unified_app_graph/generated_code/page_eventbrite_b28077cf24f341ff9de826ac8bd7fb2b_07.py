# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_07
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9.png
# step_index: 7/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top status bar
status_h = 72
status_color = (150, 150, 150)  # muted gray for status area
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)

# Subtle darker top edge and lighter bottom edge to mimic system bar shading
draw.line([(0, 0), (1440, 0)], fill=(120, 120, 120), width=2)
draw.line([(0, status_h-1), (1440, status_h-1)], fill=(200, 200, 200), width=1)

# Header area (keeps overall white but provides a clear region under the status bar)
header_top = status_h
header_bottom = 420
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Blue underline under the header/title (accent line)
underline_y = 393  # placed just under the detected title area
underline_margin = 48
draw.line([(underline_margin, underline_y), (1440 - underline_margin, underline_y)],
          fill=(35, 88, 255), width=4)

# Light divider line below the underline to give subtle separation
draw.line([(underline_margin, underline_y + 6), (1440 - underline_margin, underline_y + 6)],
          fill=(230, 230, 235), width=1)

# Section card background for the "Nearby / Current location" item
card_x0 = 36
card_x1 = 1440 - 36
card_y0 = 450
card_y1 = 579
card_radius = 16
card_fill = (250, 251, 253)  # very slight off-white / cool tint
card_outline = (230, 232, 240)
draw.rounded_rectangle([(card_x0, card_y0), (card_x1, card_y1)],
                       radius=card_radius, fill=card_fill, outline=card_outline, width=1)

# Subtle shadow under the card (soft single-line shadow to suggest elevation)
shadow_y = card_y1 + 6
draw.line([(card_x0 + 6, shadow_y), (card_x1 - 6, shadow_y)], fill=(240, 241, 245), width=2)

# Thin horizontal separator further down the page to separate list area from content region
sep_y = card_y1 + 120
draw.line([(36, sep_y), (1440 - 36, sep_y)], fill=(245, 246, 248), width=1)

# Large empty content area background (keeps white but adds a very faint center vignette area)
content_top = sep_y + 20
content_bottom = 2760
# subtle pale vignette circle to hint content region where posts would appear
center_x = 1440 // 2
center_y = (content_top + content_bottom) // 2
vignette_radius = 180
for i in range(6):
    alpha = 6 - i
    color_val = 255 - (i * 2)
    draw.ellipse([(center_x - vignette_radius - i*12, center_y - vignette_radius - i*12),
                  (center_x + vignette_radius + i*12, center_y + vignette_radius + i*12)],
                 outline=(color_val, color_val, color_val), width=1)

# Final subtle bottom footer divider
draw.line([(0, 2950), (1440, 2950)], fill=(245, 246, 248), width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 45, 69)
    canvas.paste(_c0, (1156, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1156, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/01_icon_4.44.png
try:
    _c1 = get_crop(1, 168, 168)
    canvas.paste(_c1, (0, 72), _c1)
except Exception:
    pass
layout["4.44"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/02_icon_4.44.png
try:
    _c2 = get_crop(2, 62, 65)
    canvas.paste(_c2, (179, 1), _c2)
except Exception:
    pass
layout["4.44"] = [179, 1, 241, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/03_icon_4.44.png
try:
    _c3 = get_crop(3, 61, 67)
    canvas.paste(_c3, (113, 0), _c3)
except Exception:
    pass
layout["4.44"] = [113, 0, 174, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 95, 65)
    canvas.paste(_c4, (1215, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1215, 0, 1310, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 79, 85)
    canvas.paste(_c5, (1314, 291), _c5)
except Exception:
    pass
layout["icon_5"] = [1314, 291, 1393, 376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 62, 62)
    canvas.paste(_c6, (309, 2), _c6)
except Exception:
    pass
layout["icon_6"] = [309, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 49, 57)
    canvas.paste(_c7, (249, 5), _c7)
except Exception:
    pass
layout["icon_7"] = [249, 5, 298, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 45, 58)
    canvas.paste(_c8, (1323, 3), _c8)
except Exception:
    pass
layout["icon_8"] = [1323, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/09_icon_4.44.png
try:
    _c9 = get_crop(9, 93, 64)
    canvas.paste(_c9, (15, 1), _c9)
except Exception:
    pass
layout["4.44"] = [15, 1, 108, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 51, 67)
    canvas.paste(_c10, (383, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [383, 0, 434, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/11_text_Washington.png
try:
    _c11 = get_crop(11, 1344, 129)
    canvas.paste(_c11, (48, 264), _c11)
except Exception:
    pass
layout["Washington"] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/12_text_Nearby.png
try:
    _c12 = get_crop(12, 415, 114)
    canvas.paste(_c12, (48, 465), _c12)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/13_text_Current_location.png
try:
    _c13 = get_crop(13, 415, 114)
    canvas.paste(_c13, (48, 465), _c13)
except Exception:
    pass
layout["Current_location"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_07_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-9/14_text_Loading.png
try:
    _c14 = get_crop(14, 156, 55)
    canvas.paste(_c14, (641, 1970), _c14)
except Exception:
    pass
layout["Loading"] = [641, 1970, 797, 2025]
