# page_id: page_eventbrite_eb32c51543d749539b68e6c61ff72fb8_03
# screenshot: 2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5.png
# step_index: 3/19
# task: Open Eventbrite. Set the city to San Francisco. Filter for events occurring between May 1st and May 15th under the category 'Music'. Select the first event and check the pricing options available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas and draw are provided (canvas: PIL.Image 1440x2960 RGB, draw: PIL.ImageDraw)
# Fonts available: font_sm, font_md, font_lg, font_xl

# Clear / ensure background
draw.rectangle([0, 0, 1440, 2960], fill=(255, 255, 255))

# --- Status bar ---
# Top status bar background (approx 0..80px)
status_bar_h = 80
draw.rectangle([0, 0, 1440, status_bar_h], fill=(200, 200, 200))

# subtle top highlight line to mimic device bezel
draw.line([0, status_bar_h-1, 1440, status_bar_h-1], fill=(220, 220, 220))

# --- Header area ---
# Header container below status bar (keeps white but add subtle separation shadow)
header_top = status_bar_h
header_bottom = 220
draw.rectangle([0, header_top, 1440, header_bottom], fill=(255, 255, 255))
# faint divider under header
draw.line([32, header_bottom, 1440-32, header_bottom], fill=(235, 235, 238), width=1)

# --- "Find events" input band background (structural only, not drawing text or underline) ---
# Reserve an unobtrusive area for the input region (keeps overall white but add a tiny top accent)
input_band_top = 240
input_band_bottom = 420
# very faint cool tint to show input band region without overlapping detected text
draw.rectangle([24, input_band_top, 1440-24, input_band_bottom], fill=(255, 255, 255))
# subtle bottom separator for the band
draw.line([24, input_band_bottom, 1440-24, input_band_bottom], fill=(230, 235, 255), width=2)

# --- Dual option row card (Nearby / Online events) ---
# This is the rounded background card behind the two option items.
card1_top = 430
card1_bottom = 600
card1_left = 24
card1_right = 1440 - 24
card_radius = 18
draw.rounded_rectangle([card1_left, card1_top, card1_right, card1_bottom],
                       radius=card_radius, fill=(250, 251, 255), outline=(235, 239, 255))

# vertical divider between the two option columns (purely structural)
# Position divider between the two detected element groups (approx center of available row)
divider_x = 480
draw.line([divider_x, card1_top+16, divider_x, card1_bottom-16], fill=(235, 240, 255), width=1)

# subtle inner separators to suggest item grouping
draw.line([card1_left+16, card1_bottom-1, card1_right-16, card1_bottom-1], fill=(245,245,247), width=1)

# --- Browsing location card (Los Angeles section background) ---
loc_top = 720
loc_bottom = 960
loc_left = 24
loc_right = 1440 - 24
draw.rounded_rectangle([loc_left, loc_top, loc_right, loc_bottom],
                       radius=16, fill=(255, 255, 255), outline=(242, 242, 246))

# subtle accent stroke on the left edge to indicate active section (very soft)
accent_w = 6
draw.rectangle([loc_left, loc_top+12, loc_left+accent_w, loc_bottom-12], fill=(245, 245, 250))

# faint dividing line between browsing header and location content area
draw.line([loc_left+16, loc_top+72, loc_right-16, loc_top+72], fill=(245,245,247), width=1)

# --- Subtle large-area separators to structure the page ---
# a faint long divider above the lower content area
draw.line([36, 1000, 1440-36, 1000], fill=(245,245,247), width=1)

# --- Bottom large content background (empty content region) ---
# Slightly off-white background for the large scrollable area below content to give depth
draw.rectangle([0, 1020, 1440, 2960], fill=(255, 255, 255))

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 103, 105)
    canvas.paste(_c0, (1290, 835), _c0)
except Exception:
    pass
layout["icon_0"] = [1290, 835, 1393, 940]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/01_icon_7.47.png
try:
    _c1 = get_crop(1, 58, 63)
    canvas.paste(_c1, (115, 2), _c1)
except Exception:
    pass
layout["7.47"] = [115, 2, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/02_icon_7.47.png
try:
    _c2 = get_crop(2, 60, 63)
    canvas.paste(_c2, (180, 2), _c2)
except Exception:
    pass
layout["7.47"] = [180, 2, 240, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 60, 62)
    canvas.paste(_c3, (310, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [310, 3, 370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/04_icon_7.47.png
try:
    _c4 = get_crop(4, 168, 168)
    canvas.paste(_c4, (0, 72), _c4)
except Exception:
    pass
layout["7.47"] = [0, 72, 168, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 50, 59)
    canvas.paste(_c5, (248, 5), _c5)
except Exception:
    pass
layout["icon_5"] = [248, 5, 298, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 64, 64)
    canvas.paste(_c6, (1212, 1), _c6)
except Exception:
    pass
layout["icon_6"] = [1212, 1, 1276, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 46, 58)
    canvas.paste(_c7, (1322, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1322, 3, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 49, 62)
    canvas.paste(_c8, (1264, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [1264, 1, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/09_icon_7.47.png
try:
    _c9 = get_crop(9, 91, 64)
    canvas.paste(_c9, (16, 1), _c9)
except Exception:
    pass
layout["7.47"] = [16, 1, 107, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 52, 65)
    canvas.paste(_c10, (382, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [382, 1, 434, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/11_icon_Nearby.png
try:
    _c11 = get_crop(11, 415, 114)
    canvas.paste(_c11, (48, 465), _c11)
except Exception:
    pass
layout["Nearby"] = [48, 465, 463, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/12_text_FFind_events_in..png
try:
    _c12 = get_crop(12, 1344, 129)
    canvas.paste(_c12, (48, 264), _c12)
except Exception:
    pass
layout["FFind_events_in."] = [48, 264, 1392, 393]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/13_text_Online_events.png
try:
    _c13 = get_crop(13, 452, 114)
    canvas.paste(_c13, (511, 465), _c13)
except Exception:
    pass
layout["Online_events"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/14_text_Virtual_attendance.png
try:
    _c14 = get_crop(14, 452, 114)
    canvas.paste(_c14, (511, 465), _c14)
except Exception:
    pass
layout["Virtual_attendance"] = [511, 465, 963, 579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/15_text_Browsing_in.png
try:
    _c15 = get_crop(15, 228, 55)
    canvas.paste(_c15, (44, 742), _c15)
except Exception:
    pass
layout["Browsing_in"] = [44, 742, 272, 797]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/eb32c51543d749539b68e6c61ff72fb8/step_03_2024_4_23_19_46_eb32c51543d749539b68e6c61ff72fb8-5/16_text_Los_Angeles.png
try:
    _c16 = get_crop(16, 1440, 138)
    canvas.paste(_c16, (0, 816), _c16)
except Exception:
    pass
layout["Los_Angeles"] = [0, 816, 1440, 954]
