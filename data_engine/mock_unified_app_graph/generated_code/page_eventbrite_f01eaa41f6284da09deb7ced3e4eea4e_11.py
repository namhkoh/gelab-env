# page_id: page_eventbrite_f01eaa41f6284da09deb7ced3e4eea4e_11
# screenshot: 2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13.png
# step_index: 11/11
# task: Open Eventbrite. Check out 'Sports' events. Apply filters for events happening this week. Select the first event. Check similar events and add the first similar event to favorite.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background fill (dominant off-white)
draw.rectangle([(0, 0), canvas.size], fill="#FAFAFB")

# Top status bar (system area)
status_h = 84
draw.rectangle([(0, 0), (1440, status_h)], fill="#ADB3AE")

# Subtle divider under status bar
draw.line([(0, status_h), (1440, status_h)], fill="#9aa09a", width=1)

# Snackbar / confirmation banner under status bar
snack_top = status_h + 12
snack_bottom = snack_top + 108
draw.rounded_rectangle([(16, snack_top), (1424, snack_bottom)], radius=8, fill="#E9F6EE")
# subtle bottom divider for snackbar
draw.line([(16, snack_bottom), (1424, snack_bottom)], fill="#D3E8DA", width=1)

# Area for chips/tags (clear background; we add faint section guide below)
chips_top = snack_bottom + 24
chips_bottom = chips_top + 220
# keep chips area mostly white - add a faint horizontal rule at lower edge
draw.line([(32, chips_bottom), (1408, chips_bottom)], fill="#E9E9EA", width=2)

# Thin separator above contact/report area
sep1_y = 760
draw.line([(32, sep1_y), (1408, sep1_y)], fill="#EAEAEA", width=1)

# Divider below contact/report area
sep2_y = 820
draw.line([(32, sep2_y), (1408, sep2_y)], fill="#ECECEC", width=1)

# "More like this" section background hint (we won't draw text)
more_top = 920
more_bottom = 1120
# Slight accent space (no text, just subtle background change)
draw.rectangle([(0, more_top), (1440, more_bottom)], fill="#FAFAFB")

# Event list separators and subtle row backgrounds based on detected event group positions
# First event row area (around pos y=1187)
ev1_top = 1160
ev1_bottom = ev1_top + 420  # leave room
draw.rectangle([(0, ev1_top), (1440, ev1_bottom)], fill="#FFFFFF")
draw.line([(32, ev1_bottom), (1408, ev1_bottom)], fill="#F0F0F0", width=2)

# Second event row area (around pos y=1670)
ev2_top = 1640
ev2_bottom = ev2_top + 380
draw.rectangle([(0, ev2_top), (1440, ev2_bottom)], fill="#FFFFFF")
draw.line([(32, ev2_bottom), (1408, ev2_bottom)], fill="#F0F0F0", width=2)

# Third event row area (around pos y=2092)
ev3_top = 2070
ev3_bottom = ev3_top + 300
draw.rectangle([(0, ev3_top), (1440, ev3_bottom)], fill="#FFFFFF")
draw.line([(32, ev3_bottom), (1408, ev3_bottom)], fill="#F0F0F0", width=2)

# Subtle left margin guide (do not draw thumbnails or text)
draw.line([(48, 900), (48, 2400)], fill="#F5F5F6", width=2)

# Floating ticket/reservation card (rounded white card with blue outline)
card_top = 2320
card_bottom = 2660
card_left = 40
card_right = 1400
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)], radius=28, fill="#FFFFFF", outline="#2B65FF", width=8)

# Inner horizontal divider inside the ticket card (separates title from price/controls)
inner_div_y = card_top + 120
draw.line([(card_left + 28, inner_div_y), (card_right - 28, inner_div_y)], fill="#ECEFF6", width=2)

# Small subtle shadow above the Reserve button area
shadow_top = card_bottom + 20
shadow_bottom = shadow_top + 28
draw.rectangle([(0, shadow_top), (1440, shadow_bottom)], fill="#F2F2F2")

# Reserve button area background strip (leave button itself to be pasted)
reserve_strip_top = 2748
reserve_strip_bottom = 2960
draw.rectangle([(0, reserve_strip_top), (1440, reserve_strip_bottom)], fill="#FFFFFF")

# Final subtle bottom safe-area line
draw.line([(32, reserve_strip_top), (1408, reserve_strip_top)], fill="#E8E8E8", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/00_icon_basement.png
try:
    _c0 = get_crop(0, 259, 144)
    canvas.paste(_c0, (641, 317), _c0)
except Exception:
    pass
layout["basement"] = [641, 317, 900, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/01_icon_backpacking.png
try:
    _c1 = get_crop(1, 307, 144)
    canvas.paste(_c1, (286, 317), _c1)
except Exception:
    pass
layout["backpacking"] = [286, 317, 593, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/02_icon_clinic.png
try:
    _c2 = get_crop(2, 169, 144)
    canvas.paste(_c2, (948, 317), _c2)
except Exception:
    pass
layout["clinic"] = [948, 317, 1117, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/03_icon_sports.png
try:
    _c3 = get_crop(3, 190, 144)
    canvas.paste(_c3, (48, 317), _c3)
except Exception:
    pass
layout["sports"] = [48, 317, 238, 461]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/04_icon_berkeley.png
try:
    _c4 = get_crop(4, 230, 144)
    canvas.paste(_c4, (48, 492), _c4)
except Exception:
    pass
layout["berkeley"] = [48, 492, 278, 636]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/05_icon_Reserve_a_spot.png
try:
    _c5 = get_crop(5, 1296, 132)
    canvas.paste(_c5, (72, 2756), _c5)
except Exception:
    pass
layout["Reserve_a_spot"] = [72, 2756, 1368, 2888]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/06_icon_Decrease.png
try:
    _c6 = get_crop(6, 99, 96)
    canvas.paste(_c6, (996, 2444), _c6)
except Exception:
    pass
layout["Decrease"] = [996, 2444, 1095, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/07_icon_Strategies_to_Wow_Win_More_Business.png
try:
    _c7 = get_crop(7, 1344, 387)
    canvas.paste(_c7, (48, 1187), _c7)
except Exception:
    pass
layout["Strategies_to_Wow_&_Win_M"] = [48, 1187, 1392, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/08_icon_Increase.png
try:
    _c8 = get_crop(8, 96, 96)
    canvas.paste(_c8, (1224, 2444), _c8)
except Exception:
    pass
layout["Increase"] = [1224, 2444, 1320, 2540]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/09_icon_Basement_Berkeleyl.png
try:
    _c9 = get_crop(9, 1344, 326)
    canvas.paste(_c9, (48, 1670), _c9)
except Exception:
    pass
layout["Basement_Berkeleyl"] = [48, 1670, 1392, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 91, 103)
    canvas.paste(_c10, (1109, 2441), _c10)
except Exception:
    pass
layout["icon_10"] = [1109, 2441, 1200, 2544]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/11_icon_Dismiss_notification.png
try:
    _c11 = get_crop(11, 142, 142)
    canvas.paste(_c11, (1251, 97), _c11)
except Exception:
    pass
layout["Dismiss_notification"] = [1251, 97, 1393, 239]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/12_icon_closed_onllne_therapeutic_art_group_t0.png
try:
    _c12 = get_crop(12, 1344, 232)
    canvas.paste(_c12, (48, 2092), _c12)
except Exception:
    pass
layout["closed_onllne_therapeutic"] = [48, 2092, 1392, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/13_icon_4.36.png
try:
    _c13 = get_crop(13, 60, 64)
    canvas.paste(_c13, (113, 1), _c13)
except Exception:
    pass
layout["4.36"] = [113, 1, 173, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/14_icon_4.36.png
try:
    _c14 = get_crop(14, 58, 62)
    canvas.paste(_c14, (181, 1), _c14)
except Exception:
    pass
layout["4.36"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 57, 59)
    canvas.paste(_c15, (312, 3), _c15)
except Exception:
    pass
layout["icon_15"] = [312, 3, 369, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/16_icon_Constellations_Counselling.png
try:
    _c16 = get_crop(16, 1344, 232)
    canvas.paste(_c16, (48, 2092), _c16)
except Exception:
    pass
layout["Constellations_Counsellin"] = [48, 2092, 1392, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/17_icon_Like.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1284, 1888), _c17)
except Exception:
    pass
layout["Like"] = [1284, 1888, 1428, 2032]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 47, 57)
    canvas.paste(_c18, (251, 4), _c18)
except Exception:
    pass
layout["icon_18"] = [251, 4, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/19_icon_icon_19.png
try:
    _c19 = get_crop(19, 52, 63)
    canvas.paste(_c19, (1319, 0), _c19)
except Exception:
    pass
layout["icon_19"] = [1319, 0, 1371, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 99, 61)
    canvas.paste(_c20, (1212, 1), _c20)
except Exception:
    pass
layout["icon_20"] = [1212, 1, 1311, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/21_icon_Like.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1284, 1405), _c21)
except Exception:
    pass
layout["Like"] = [1284, 1405, 1428, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/22_icon_Bs.png
try:
    _c22 = get_crop(22, 1344, 326)
    canvas.paste(_c22, (48, 1670), _c22)
except Exception:
    pass
layout["Bs"] = [48, 1670, 1392, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/23_icon_icon_23.png
try:
    _c23 = get_crop(23, 50, 59)
    canvas.paste(_c23, (383, 3), _c23)
except Exception:
    pass
layout["icon_23"] = [383, 3, 433, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/24_icon_Share.png
try:
    _c24 = get_crop(24, 120, 144)
    canvas.paste(_c24, (1164, 1888), _c24)
except Exception:
    pass
layout["Share"] = [1164, 1888, 1284, 2032]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/25_icon_Share.png
try:
    _c25 = get_crop(25, 120, 144)
    canvas.paste(_c25, (1164, 1405), _c25)
except Exception:
    pass
layout["Share"] = [1164, 1405, 1284, 1549]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/26_icon_Free.png
try:
    _c26 = get_crop(26, 139, 123)
    canvas.paste(_c26, (97, 2566), _c26)
except Exception:
    pass
layout["Free"] = [97, 2566, 236, 2689]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/27_icon_Oakland_Ballers_Meet_Greet_at_Sports.png
try:
    _c27 = get_crop(27, 1344, 326)
    canvas.paste(_c27, (48, 1670), _c27)
except Exception:
    pass
layout["Oakland_Ballers_Meet_&_Gr"] = [48, 1670, 1392, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/28_icon_Free.png
try:
    _c28 = get_crop(28, 75, 72)
    canvas.paste(_c28, (249, 2588), _c28)
except Exception:
    pass
layout["Free"] = [249, 2588, 324, 2660]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/29_icon_4.36.png
try:
    _c29 = get_crop(29, 93, 62)
    canvas.paste(_c29, (14, 2), _c29)
except Exception:
    pass
layout["4.36"] = [14, 2, 107, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 73, 83)
    canvas.paste(_c30, (1319, 2274), _c30)
except Exception:
    pass
layout["icon_30"] = [1319, 2274, 1392, 2357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/31_icon_sGrow.png
try:
    _c31 = get_crop(31, 128, 62)
    canvas.paste(_c31, (401, 1380), _c31)
except Exception:
    pass
layout["sGrow"] = [401, 1380, 529, 1442]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/32_icon_Report_event.png
try:
    _c32 = get_crop(32, 246, 144)
    canvas.paste(_c32, (829, 766), _c32)
except Exception:
    pass
layout["Report_event"] = [829, 766, 1075, 910]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/33_icon_4.36.png
try:
    _c33 = get_crop(33, 144, 144)
    canvas.paste(_c33, (36, 108), _c33)
except Exception:
    pass
layout["4.36"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/34_icon_Realtors_Refine_Client_Appreciation.png
try:
    _c34 = get_crop(34, 1344, 387)
    canvas.paste(_c34, (48, 1187), _c34)
except Exception:
    pass
layout["Realtors:_Refine_Client_A"] = [48, 1187, 1392, 1574]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/35_icon_Contact_the.png
try:
    _c35 = get_crop(35, 416, 144)
    canvas.paste(_c35, (365, 766), _c35)
except Exception:
    pass
layout["Contact_the"] = [365, 766, 781, 910]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/36_icon_Art_for_Grief_and_Loss.png
try:
    _c36 = get_crop(36, 1344, 232)
    canvas.paste(_c36, (48, 2092), _c36)
except Exception:
    pass
layout["Art_for_Grief_and_Loss"] = [48, 2092, 1392, 2324]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/37_icon_Sports_Basement.png
try:
    _c37 = get_crop(37, 1344, 326)
    canvas.paste(_c37, (48, 1670), _c37)
except Exception:
    pass
layout["Sports_Basement"] = [48, 1670, 1392, 1996]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/38_text_We_ve_added_the_event_to_your_shortlist.png
try:
    _c38 = get_crop(38, 144, 144)
    canvas.paste(_c38, (36, 108), _c38)
except Exception:
    pass
layout["We've_added_the_event_to_"] = [36, 108, 180, 252]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/39_text_More_like_this.png
try:
    _c39 = get_crop(39, 367, 61)
    canvas.paste(_c39, (45, 1072), _c39)
except Exception:
    pass
layout["More_like_this"] = [45, 1072, 412, 1133]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f01eaa41f6284da09deb7ced3e4eea4e/step_11_2024_4_24_16_35_f01eaa41f6284da09deb7ced3e4eea4e-13/40_text_General_Admission.png
try:
    _c40 = get_crop(40, 75, 72)
    canvas.paste(_c40, (249, 2588), _c40)
except Exception:
    pass
layout["General_Admission"] = [249, 2588, 324, 2660]
